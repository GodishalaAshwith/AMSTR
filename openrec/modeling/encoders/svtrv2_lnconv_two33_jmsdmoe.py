import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from openrec.modeling.common import DropPath, Identity, Mlp


# ============= 1. Jumble Module =============
class JumbleModule(nn.Module):
    def __init__(self, patch_size=4, shuffle_ratio=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.shuffle_ratio = shuffle_ratio

    def forward(self, x):
        if not self.training or self.shuffle_ratio == 0.0:
            return x
        B, C, H, W = x.shape
        ps = self.patch_size
        assert H % ps == 0 and W % ps == 0, f"Image dims must be divisible by {ps}"
        x = x.view(B, C, H // ps, ps, W // ps, ps)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, C, num_h, num_w, ps, ps)
        num_h, num_w = H // ps, W // ps
        total_patches = num_h * num_w
        x = x.view(B, C, total_patches, ps, ps)
        num_shuffle = int(total_patches * self.shuffle_ratio)
        if num_shuffle == 0:
            return x.view(B, C, num_h, num_w, ps, ps).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
        for b in range(B):
            indices = torch.randperm(total_patches, device=x.device)
            shuffle_idx = indices[:num_shuffle]
            keep_idx = indices[num_shuffle:]
            shuffled_patches = x[b, :, shuffle_idx]
            shuffled_patches = shuffled_patches[:, torch.randperm(num_shuffle, device=x.device)]
            new_patches = torch.empty_like(x[b])
            new_patches[:, keep_idx] = x[b, :, keep_idx]
            new_patches[:, shuffle_idx] = shuffled_patches
            x[b] = new_patches
        x = x.view(B, C, num_h, num_w, ps, ps)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, H, W)
        return x


# ============= 2. Self-Distillation Module =============
class SelfDistillationModule(nn.Module):
    def __init__(self, dim, stride=3, reduction_ratio=4):
        super().__init__()
        hidden_dim = dim // reduction_ratio
        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        return out


# ============= 3. MoE MLP (Soft MoE) =============
class MoEMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        num_experts=4,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.num_experts = num_experts
        self.gate = nn.Linear(in_features, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                act_layer(),
                nn.Dropout(drop),
                nn.Linear(hidden_features, out_features),
                nn.Dropout(drop)
            ) for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        B, N, C = x.shape
        gate_logits = self.gate(x)  # (B, N, E)
        gate_weights = gate_logits.softmax(dim=-1)  # soft MoE
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (B, N, C, E)
        x = (expert_outputs * gate_weights.unsqueeze(-2)).sum(dim=-1)  # (B, N, C)
        x = self.norm(x)
        return x


# ============= 4. 基础模块（略作修改以支持 MoE） =============
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
        mlp_layer=None,  # <<<<< 支持自定义 MLP
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
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class FlattenBlockRe2D(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                 drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, eps=1e-6, mlp_layer=None):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                         attn_drop, drop_path, act_layer, norm_layer, eps, mlp_layer)

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
        mlp_layer=None,  # <<<<<
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
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x


# ============= 5. Attention & 其他基础模块（保持不变） =============
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
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
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, [H, W]


class IdentitySize(nn.Module):
    def forward(self, x, sz):
        return x, sz


# ============= 6. SVTRStage with SD + MoE support =============
class SVTRStage(nn.Module):
    def __init__(self,
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
                 use_self_distill=False,
                 distill_stride=3,
                 mlp_layer=None,  # <<<<< 支持 MoE
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.blocks = nn.Sequential()
        for i in range(depth):
            if mixer[i] == 'Conv':
                self.blocks.append(
                    ConvBlock(dim=dim,
                              kernel_size=kernel_sizes[i],
                              num_heads=num_heads,
                              mlp_ratio=mlp_ratio,
                              drop=drop_rate,
                              act_layer=act,
                              drop_path=drop_path[i],
                              norm_layer=norm_layer,
                              eps=eps,
                              num_conv=num_conv[i],
                              mlp_layer=mlp_layer))
            else:
                if mixer[i] == 'Global':
                    block = Block
                elif mixer[i] == 'FGlobal':
                    block = Block
                    self.blocks.append(FlattenTranspose())
                elif mixer[i] == 'FGlobalRe2D':
                    block = FlattenBlockRe2D
                self.blocks.append(
                    block(
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
                    ))

        self.use_self_distill = use_self_distill
        if use_self_distill:
            self.self_distill = SelfDistillationModule(dim=dim, stride=distill_stride)
        else:
            self.self_distill = None

        if downsample:
            if mixer[-1] in ['Conv', 'FGlobalRe2D']:
                self.downsample = SubSample2D(dim, out_dim, stride=sub_k)
            else:
                self.downsample = SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = IdentitySize()

    def forward(self, x, sz):
        for blk in self.blocks:
            x = blk(x)
        if self.use_self_distill:
            if len(x.shape) == 3:
                B, N, C = x.shape
                H, W = sz
                x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.self_distill(x)
            H, W = x.shape[2], x.shape[3]
            sz = [H, W]
        x, sz = self.downsample(x, sz)
        return x, sz


# ============= 7. 其他辅助模块（保持不变） =============
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
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        return x, sz


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 bias=False, groups=1, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ============= 8. FINAL MODEL: SVTRv2LNConvTwo33JMSDMoE =============
class SVTRv2LNConvTwo33JMSDMoE(nn.Module):
    """
    SVTRv2 with:
      - Jumble Module (JM) before patch embedding
      - Self-Distillation Module (SD) after each stage
      - MoE-style MLP (MoE) in every block
    """

    def __init__(self,
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
                 # >>>>> JM 参数
                 use_jumble=True,
                 jumble_patch_size=4,
                 jumble_ratio=0.1,
                 # >>>>> SD 参数
                 use_self_distill=True,
                 distill_stride=3,
                 # >>>>> MoE 参数
                 use_moe=True,
                 moe_num_experts=4,
                 **kwargs):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        # >>>>> JM
        self.use_jumble = use_jumble
        if use_jumble:
            self.jumble = JumbleModule(patch_size=jumble_patch_size, shuffle_ratio=jumble_ratio)
        else:
            self.jumble = nn.Identity()

        feat_max_size = [max_sz[0] // 4, max_sz[1] // 4]
        self.pope = POPatchEmbed(in_channels=in_channels,
                                 feat_max_size=feat_max_size,
                                 embed_dim=dims[0],
                                 use_pos_embed=use_pos_embed,
                                 flatten=mixer[0][0] != 'Conv',
                                 bias=pope_bias)

        dpr = np.linspace(0, drop_path_rate, sum(depths))

        # >>>>> MoE MLP factory
        def make_moe_mlp(**kwargs):
            return MoEMLP(num_experts=moe_num_experts, **kwargs)

        mlp_layer = make_moe_mlp if use_moe else None

        # >>>>> Build stages with SD + MoE
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
                use_self_distill=use_self_distill,
                distill_stride=distill_stride,
                mlp_layer=mlp_layer,
            )
            self.stages.append(stage)

        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.stages.append(LastStage(self.num_features, out_channels, last_drop))
        if feat2d:
            self.stages.append(Feat2D())
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
        return {'patch_embed', 'downsample', 'pos_embed'}

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.flatten(0, 1)
        # >>>>> JM
        x = self.jumble(x)
        # <<<<<
        x, sz = self.pope(x)
        for stage in self.stages:
            x, sz = stage(x, sz)
        return x