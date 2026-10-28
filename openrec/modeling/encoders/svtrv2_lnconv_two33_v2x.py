import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from openrec.modeling.common import DropPath, Identity, Mlp


# ==============================================================================
# 1. Jumble Module (JM) - Section 3.2, Equations 1 & 2
# ==============================================================================
class JumbleModule(nn.Module):
    """
    Jumble Module (JM):
    Divides input image I into N non-overlapping patches of size p x p (p=4 default),
    randomly selects a subset S with shuffle ratio P (default 0.1), and permutes
    patches within S.
    Only active during training; identity pass-through during inference.
    """
    def __init__(self, patch_size=4, shuffle_ratio=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.shuffle_ratio = shuffle_ratio

    def forward(self, x):
        if not self.training or self.shuffle_ratio <= 0.0:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            return x

        num_h, num_w = H // p, W // p
        total_patches = num_h * num_w
        num_shuffle = int(total_patches * self.shuffle_ratio)
        if num_shuffle <= 1:
            return x

        # Reshape into patches: (B, C, num_h, p, num_w, p) -> (B, total_patches, C, p, p)
        patches = x.view(B, C, num_h, p, num_w, p).permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, total_patches, C, p, p)

        # Batch-wise shuffling of selected subset S
        out_patches = patches.clone()
        for b in range(B):
            perm = torch.randperm(total_patches, device=x.device)
            shuffle_idx = perm[:num_shuffle]
            dest_perm = shuffle_idx[torch.randperm(num_shuffle, device=x.device)]
            out_patches[b, shuffle_idx] = patches[b, dest_perm]

        # Reassemble to original image shape: (B, C, H, W)
        out = out_patches.view(B, num_h, num_w, C, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
        return out.view(B, C, H, W)


# ==============================================================================
# 2. Sparse Mixture of Experts (MoE) - Section 3.4, Equations 7-9 & Equation 13
# ==============================================================================
class SparseMoEMLP(nn.Module):
    """
    Sparse Mixture of Experts MLP:
    - N parallel expert sub-networks (default 4 experts)
    - Top-1 routing (K=1) for activation sparsity (+1.3% FLOPs)
    - Auxiliary load balancing loss computation: L_load = N * sum(f_i * p_i)
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        num_experts=4,
        topk=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.num_experts = num_experts
        self.topk = topk

        # Gating network: W_g * x + b_g
        self.gate = nn.Linear(in_features, num_experts)

        # Expert networks: 2-layer FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                act_layer(),
                nn.Dropout(drop),
                nn.Linear(hidden_features, out_features),
                nn.Dropout(drop),
            ) for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(out_features)
        self.last_load_loss = torch.tensor(0.0)

    def forward(self, x):
        # x: (B, T, C)
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])  # (M, C) where M = B * T
        M, C = x_flat.shape

        # Gating logits & probabilities: g = softmax(W_g x + b_g) in fp32 for numerical stability
        gate_logits = self.gate(x_flat)  # (M, num_experts)
        gate_probs = F.softmax(gate_logits.float(), dim=-1).type_as(gate_logits)

        # Top-1 routing (or top-K)
        top_weights, top_indices = torch.topk(gate_probs, self.topk, dim=-1)  # (M, K)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute load balancing loss during training: L_load = N * sum(f_i * p_i)
        if self.training:
            # f_i: fraction of tokens dispatched to expert i (detached for stable gradients)
            expert_mask = F.one_hot(top_indices.squeeze(-1), num_classes=self.num_experts).float()
            f = expert_mask.mean(dim=0).detach()  # (num_experts,)
            p = gate_probs.mean(dim=0)            # (num_experts,)
            self.last_load_loss = self.num_experts * torch.sum(f * p)
        else:
            self.last_load_loss = torch.tensor(0.0, device=x.device)

        # Sparse dispatch & aggregation
        y_flat = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (top_indices.squeeze(-1) == i)
            if mask.any():
                selected_x = x_flat[mask]
                selected_weight = top_weights[mask]  # (K_tokens, 1)
                expert_out = self.experts[i](selected_x)
                y_flat[mask] = expert_out * selected_weight

        out = self.norm(y_flat).view(orig_shape)
        return out


# ==============================================================================
# 3. Self-Distillation Module (SDM) Projection Head - Section 3.3, Equations 3-5
# ==============================================================================
class SDMHead(nn.Module):
    """
    SDM transformation head:
    F_hat = Conv1x1(F) -> BatchNorm -> ReLU
    F_tilde = Conv3x3_non_overlap(F_hat) -> BatchNorm -> ReLU
    F_prime = Conv1x1(F_tilde) -> BatchNorm
    """
    def __init__(self, in_dim, shared_dim=128, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, shared_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(shared_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(shared_dim, shared_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(shared_dim)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(shared_dim, shared_dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(shared_dim)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out


# ==============================================================================
# 4. Standard SVTRv2 Building Blocks (with MoE Support)
# ==============================================================================
class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        return self.act(self.norm(self.conv(inputs)))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
        mlp_layer=None,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        if mlp_layer is not None:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class FlattenBlockRe2D(Block):
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
        num_conv=2,
        kernel_size=3,
        mlp_layer=None,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = nn.Sequential(*[
            nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=num_heads)
            for _ in range(num_conv)
        ])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        if mlp_layer is not None:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

    def forward(self, x):
        C, H, W = x.shape[1:]
        x = x + self.drop_path(self.mixer(x))
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x


class FlattenTranspose(nn.Module):
    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


class SubSample2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[2, 1]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x, [H, W]


class SubSample1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[2, 1]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        if len(x.shape) == 4:
            x = x.flatten(2).transpose(1, 2)
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, [H, W]


class IdentitySize(nn.Module):
    def forward(self, x, sz):
        return x, sz


class ADDPosEmbed(nn.Module):
    def __init__(self, feat_max_size=[8, 32], embed_dim=768):
        super().__init__()
        pos_embed = torch.zeros([1, feat_max_size[0] * feat_max_size[1], embed_dim])
        trunc_normal_(pos_embed, mean=0, std=0.02)
        self.pos_embed = nn.Parameter(
            pos_embed.transpose(1, 2).reshape(1, embed_dim, feat_max_size[0], feat_max_size[1]),
            requires_grad=True,
        )

    def forward(self, x):
        sz = x.shape[2:]
        x = x + self.pos_embed[:, :, :sz[0], :sz[1]]
        return x


# ==============================================================================
# Tier 1 Architecture Injection: SE-Augmented Residual Stem & 2D Pos Embed
# ==============================================================================

class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block:
    Global Average Pooling -> Linear(D -> D//r) -> ReLU -> Linear(D//r -> D) -> Sigmoid
    Dynamically weights feature channels to emphasize fine diacritic strokes and
    suppress background noise.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, _, _ = x.shape
        w = self.avg_pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class ConvResidualBlock(nn.Module):
    """
    3x3 Convolutional Residual Block:
    Conv3x3 -> BN -> GELU -> Conv3x3 -> BN + Identity shortcut -> GELU
    Deepens receptive fields to preserve high-frequency sub-glyph details.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act2 = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + residual)
        return out


class Learnable2DPosEmbed(nn.Module):
    """
    2D Learnable Positional Embedding:
    Initialized as a learnable parameter matrix of size [1, H_max, W_max, D].
    Added to patch embeddings before flattening for the transformer sequence to preserve
    explicit row-and-column coordinate layouts.
    Supports dynamic slicing and bicubic interpolation fallback for dynamic text lengths.
    """
    def __init__(self, feat_max_size=[8, 64], embed_dim=64):
        super().__init__()
        self.feat_max_size = feat_max_size
        self.embed_dim = embed_dim
        # Shape: [1, H_max, W_max, D]
        self.pos_embed = nn.Parameter(
            torch.zeros(1, feat_max_size[0], feat_max_size[1], embed_dim)
        )
        trunc_normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x_2d):
        """
        x_2d: (B, H, W, D) where D == embed_dim
        """
        B, H, W, D = x_2d.shape
        if H <= self.pos_embed.shape[1] and W <= self.pos_embed.shape[2]:
            pe = self.pos_embed[:, :H, :W, :]
        else:
            # Fallback dynamic bicubic interpolation if input dimensions exceed feat_max_size
            pe = self.pos_embed.permute(0, 3, 1, 2)  # [1, D, H_max, W_max]
            pe = F.interpolate(pe, size=(H, W), mode='bicubic', align_corners=False)
            pe = pe.permute(0, 2, 3, 1)  # [1, H, W, D]
        return x_2d + pe


class SEAugmentedStem(nn.Module):
    """
    Tier 1 SE-Augmented Residual Stem with 2D Positional Embeddings:
    1. Patch downsampling (two stride-2 convs: 3 -> D//2 -> D, 4x downsample)
    2. Two 3x3 ConvResidualBlocks
    3. Squeeze-and-Excitation (SE) channel modulation
    4. 2D Learnable Positional Embedding [1, H', W', D]
    5. Optional sequence flattening for transformer blocks
    """
    def __init__(
        self,
        in_channels=3,
        feat_max_size=[8, 64],
        embed_dim=64,
        use_2d_pos_embed=True,
        se_reduction=4,
        flatten=False,
        bias=False,
    ):
        super().__init__()
        self.flatten = flatten
        self.use_2d_pos_embed = use_2d_pos_embed

        # Initial 4x patch downsample
        self.downsample_convs = nn.Sequential(
            ConvBNLayer(in_channels, embed_dim // 2, 3, 2, 1, bias=bias, act=nn.GELU),
            ConvBNLayer(embed_dim // 2, embed_dim, 3, 2, 1, bias=bias, act=nn.GELU),
        )
        # Two 3x3 residual blocks
        self.res_block1 = ConvResidualBlock(embed_dim)
        self.res_block2 = ConvResidualBlock(embed_dim)
        # SE channel weighting
        self.se = SqueezeAndExcitation(embed_dim, reduction=se_reduction)

        # 2D Positional Embeddings
        if use_2d_pos_embed:
            self.pos_embed_2d = Learnable2DPosEmbed(feat_max_size, embed_dim)
        else:
            self.pos_embed_2d = None

    def forward(self, x):
        sz = x.shape[2:]
        H_prime, W_prime = sz[0] // 4, sz[1] // 4

        # 1. Downsampling convs: [B, 3, H, W] -> [B, D, H/4, W/4]
        x = self.downsample_convs(x)
        # 2. Two 3x3 Conv residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        # 3. Squeeze-and-Excitation channel gating
        x = self.se(x)

        # 4. 2D Positional Embeddings
        if self.pos_embed_2d is not None:
            # Permute to [B, H', W', D] to add 2D coordinates
            x_2d = x.permute(0, 2, 3, 1)
            x_2d = self.pos_embed_2d(x_2d)
            if self.flatten:
                # Flatten [B, H'*W', D] for transformer sequence
                x = x_2d.flatten(1, 2)
            else:
                # Permute back to [B, D, H', W']
                x = x_2d.permute(0, 3, 1, 2)
        else:
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)

        return x, [H_prime, W_prime]


class POPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, feat_max_size=[8, 32], embed_dim=768,
                 use_pos_embed=False, flatten=False, bias=False):
        super().__init__()
        self.patch_embed = nn.Sequential(
            ConvBNLayer(in_channels, embed_dim // 2, 3, 2, 1, bias=bias, act=nn.GELU),
            ConvBNLayer(embed_dim // 2, embed_dim, 3, 2, 1, bias=bias, act=nn.GELU),
        )
        if use_pos_embed:
            self.patch_embed.append(ADDPosEmbed(feat_max_size, embed_dim))
        if flatten:
            self.patch_embed.append(FlattenTranspose())

    def forward(self, x):
        sz = x.shape[2:]
        x = self.patch_embed(x)
        return x, [sz[0] // 4, sz[1] // 4]


class LastStage(nn.Module):
    def __init__(self, in_channels, out_channels, last_drop, out_char_num=0):
        super().__init__()
        self.last_conv = nn.Linear(in_channels, out_channels, bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop)

    def forward(self, x, sz):
        x = x.reshape(-1, sz[0], sz[1], x.shape[-1])
        x = x.mean(1)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        return x, [1, sz[1]]


class Feat2D(nn.Module):
    def forward(self, x, sz):
        if len(x.shape) == 4:
            x = x.flatten(2).transpose(1, 2)
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        return x, sz


class SVTRStage(nn.Module):
    def __init__(
        self,
        dim=64,
        out_dim=256,
        depth=3,
        mixer=['Local'] * 3,
        kernel_sizes=[3] * 3,
        sub_k=[2, 1],
        num_heads=2,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path=[0.1] * 3,
        norm_layer=nn.LayerNorm,
        act=nn.GELU,
        eps=1e-6,
        num_conv=[2] * 3,
        downsample=None,
        mlp_layer=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if mixer[i] == 'Conv':
                self.blocks.append(
                    ConvBlock(
                        dim=dim,
                        kernel_size=kernel_sizes[i],
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        act_layer=act,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        eps=eps,
                        num_conv=num_conv[i],
                        mlp_layer=mlp_layer,
                    )
                )
            else:
                if mixer[i] == 'Global':
                    block_cls = Block
                elif mixer[i] == 'FGlobal':
                    block_cls = Block
                    self.blocks.append(FlattenTranspose())
                elif mixer[i] == 'FGlobalRe2D':
                    block_cls = FlattenBlockRe2D
                self.blocks.append(
                    block_cls(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        act_layer=act,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        eps=eps,
                        mlp_layer=mlp_layer,
                    )
                )

        if downsample:
            if mixer[-1] in ['Conv', 'FGlobalRe2D']:
                self.downsample = SubSample2D(dim, out_dim, stride=sub_k)
            else:
                self.downsample = SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = IdentitySize()

    def forward_features(self, x):
        """Pass through blocks and return feature before downsampling."""
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x, sz):
        x = self.forward_features(x)
        x, sz = self.downsample(x, sz)
        return x, sz


# ==============================================================================
# 5. Full SVTRv2X Architecture (Guo et al., PLOS ONE 2026)
# ==============================================================================
class SVTRv2XEncoder(nn.Module):
    """
    SVTRv2X: Enhanced scene text recognition via self-distilled mixture-of-experts
    Built upon SVTRv2 with 3 synergistic modules:
      1. Jumble Module (JM): Patch-level spatial shuffling prior to patch embedding.
      2. Mixture-of-Experts (MoE): Top-1 Sparse MoE replacing MLPs in mixing blocks.
      3. Self-Distillation Module (SDM): Auxiliary 1x1->3x3->1x1 heads aligning Stage 1 and 2.
    """
    def __init__(
        self,
        max_sz=[32, 128],
        in_channels=3,
        out_channels=192,
        depths=[3, 6, 3],
        dims=[64, 128, 256],
        mixer=[['Conv'] * 3, ['Conv'] * 3 + ['Global'] * 3, ['Global'] * 3],
        use_pos_embed=True,
        sub_k=[[1, 1], [2, 1], [1, 1]],
        num_heads=[2, 4, 8],
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        last_drop=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act=nn.GELU,
        last_stage=False,
        feat2d=False,
        eps=1e-6,
        num_convs=[[2] * 3, [2] * 3 + [3] * 3, [3] * 3],
        kernel_sizes=[[3] * 3, [3] * 3 + [3] * 3, [3] * 3],
        pope_bias=False,
        # >>> Tier 1 Options (Run 2)
        use_se_stem=False,
        use_2d_pos_embed=False,
        se_reduction=4,
        # >>> JM Options
        use_jumble=True,
        jumble_patch_size=4,
        jumble_ratio=0.1,
        # >>> MoE Options
        use_moe=True,
        moe_num_experts=4,
        moe_topk=1,
        # >>> SDM Options
        use_self_distill=True,
        sdm_shared_dim=128,
        **kwargs,
    ):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]
        self.use_self_distill = use_self_distill
        self.use_moe = use_moe
        self.use_se_stem = use_se_stem
        self.use_2d_pos_embed = use_2d_pos_embed

        # 1. Jumble Module
        self.use_jumble = use_jumble
        if use_jumble:
            self.jumble = JumbleModule(patch_size=jumble_patch_size, shuffle_ratio=jumble_ratio)
        else:
            self.jumble = nn.Identity()

        # Patch Embedding
        feat_max_size = [max_sz[0] // 4, max(64, max_sz[1] // 4)]
        if use_se_stem:
            self.pope = SEAugmentedStem(
                in_channels=in_channels,
                feat_max_size=feat_max_size,
                embed_dim=dims[0],
                use_2d_pos_embed=use_2d_pos_embed,
                se_reduction=se_reduction,
                flatten=mixer[0][0] != 'Conv',
                bias=pope_bias,
            )
        else:
            self.pope = POPatchEmbed(
                in_channels=in_channels,
                feat_max_size=[max_sz[0] // 4, max_sz[1] // 4],
                embed_dim=dims[0],
                use_pos_embed=use_pos_embed,
                flatten=mixer[0][0] != 'Conv',
                bias=pope_bias,
            )

        dpr = np.linspace(0, drop_path_rate, sum(depths))

        # MLP layer factory (MoE vs standard Dense MLP)
        def make_moe_mlp(**m_kwargs):
            return SparseMoEMLP(num_experts=moe_num_experts, topk=moe_topk, **m_kwargs)

        mlp_layer = make_moe_mlp if use_moe else None

        # Build Stages
        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            stage = SVTRStage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                kernel_sizes=kernel_sizes[i_stage]
                if len(kernel_sizes[i_stage]) == len(mixer[i_stage]) else [3] * len(mixer[i_stage]),
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=False if i_stage == num_stages - 1 else True,
                eps=eps,
                num_conv=num_convs[i_stage] if len(num_convs[i_stage]) == len(mixer[i_stage]) else [2] * len(mixer[i_stage]),
                mlp_layer=mlp_layer,
            )
            self.stages.append(stage)

        # 3. Self-Distillation Heads for Stage 1 and Stage 2
        if use_self_distill:
            # Stage 1 feature resolution has height 8; Stage 2 has height 4 (stride=[2, 1] downsample)
            # Use stride=(2, 1) in stage 1 SDM head so spatial dimensions match stage 2
            self.sdm_head1 = SDMHead(in_dim=dims[0], shared_dim=sdm_shared_dim, stride=(2, 1))
            self.sdm_head2 = SDMHead(in_dim=dims[1], shared_dim=sdm_shared_dim, stride=(1, 1))
        else:
            self.sdm_head1 = None
            self.sdm_head2 = None

        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.stages.append(LastStage(self.num_features, out_channels, last_drop))
        if feat2d:
            self.stages.append(Feat2D())

        self.aux_dict = {}
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_embed', 'downsample', 'pos_embed', 'pos_embed_2d'}

    def get_moe_load_loss(self):
        """Aggregate auxiliary load balancing loss across all MoE blocks."""
        if not self.use_moe:
            return torch.tensor(0.0)
        total_loss = 0.0
        count = 0
        for m in self.modules():
            if isinstance(m, SparseMoEMLP):
                total_loss = total_loss + m.last_load_loss
                count += 1
        return total_loss / max(count, 1)

    def _ensure_2d_feat(self, feat, sz):
        """Convert token sequence (B, N, C) back to (B, C, H, W) for conv-based SDM."""
        if len(feat.shape) == 4:
            return feat
        B, N, C = feat.shape
        H, W = sz
        return feat.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.flatten(0, 1)

        # 1. Jumble Module (active only in training)
        x = self.jumble(x)

        # Patch Embedding
        x, sz = self.pope(x)

        # Stage 1
        f1_raw = self.stages[0].forward_features(x)
        f1_sz = sz
        x, sz = self.stages[0].downsample(f1_raw, sz)

        # Stage 2
        f2_raw = self.stages[1].forward_features(x)
        f2_sz = sz
        x, sz = self.stages[1].downsample(f2_raw, sz)

        # Remaining Stages (Stage 3, optional LastStage/Feat2D)
        for stage in self.stages[2:]:
            x, sz = stage(x, sz)

        # Auxiliary outputs for training (SDM + MoE load loss)
        if self.training:
            self.aux_dict = {}
            if self.use_self_distill:
                f1_2d = self._ensure_2d_feat(f1_raw, f1_sz)
                f2_2d = self._ensure_2d_feat(f2_raw, f2_sz)
                proj1 = self.sdm_head1(f1_2d)
                proj2 = self.sdm_head2(f2_2d)
                if proj1.shape[2:] != proj2.shape[2:]:
                    proj1 = F.adaptive_avg_pool2d(proj1, proj2.shape[2:])
                self.aux_dict['sdm_f1'] = proj1
                self.aux_dict['sdm_f2'] = proj2
            if self.use_moe:
                self.aux_dict['moe_loss'] = self.get_moe_load_loss()
        else:
            self.aux_dict = {}

        return x


# Alias for backward compatibility
SVTRv2LNConvTwo33JMSDMoE = SVTRv2XEncoder


class SVTRv2LNConvTwo33JM(SVTRv2XEncoder):
    """Ablation: Jumble Module alone"""
    def __init__(self, **kwargs):
        kwargs.setdefault('use_jumble', True)
        kwargs.setdefault('use_moe', False)
        kwargs.setdefault('use_self_distill', False)
        super().__init__(**kwargs)


class SVTRv2LNConvTwo33SD(SVTRv2XEncoder):
    """Ablation: Self-Distillation Module alone"""
    def __init__(self, **kwargs):
        kwargs.setdefault('use_jumble', False)
        kwargs.setdefault('use_moe', False)
        kwargs.setdefault('use_self_distill', True)
        super().__init__(**kwargs)


class SVTRv2LNConvTwo33MoE(SVTRv2XEncoder):
    """Ablation: Mixture of Experts alone"""
    def __init__(self, **kwargs):
        kwargs.setdefault('use_jumble', False)
        kwargs.setdefault('use_moe', True)
        kwargs.setdefault('use_self_distill', False)
        super().__init__(**kwargs)


class SVTRv2LNConvTwo33JMSD(SVTRv2XEncoder):
    """Ablation: Jumble Module + Self-Distillation"""
    def __init__(self, **kwargs):
        kwargs.setdefault('use_jumble', True)
        kwargs.setdefault('use_moe', False)
        kwargs.setdefault('use_self_distill', True)
        super().__init__(**kwargs)


class SVTRv2X_Tier1(SVTRv2XEncoder):
    """Run 2: Tier 1 Architecture Injection (SE-Augmented Stem + 2D Positional Embeddings)"""
    def __init__(self, **kwargs):
        kwargs.setdefault('use_se_stem', True)
        kwargs.setdefault('use_2d_pos_embed', True)
        kwargs.setdefault('se_reduction', 4)
        super().__init__(**kwargs)


