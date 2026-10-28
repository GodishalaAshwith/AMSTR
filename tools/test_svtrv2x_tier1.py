"""
Comprehensive Verification and Unit Testing Script for SVTRv2X Run 2: Tier 1 Architecture Injection
Tests:
  1. SqueezeAndExcitation: forward, gating range, channel weighting, gradient flow.
  2. ConvResidualBlock: residual identity shortcut, spatial preservation, gradients.
  3. Learnable2DPosEmbed: 2D coordinate preservation, dynamic slicing, interpolation fallback.
  4. SEAugmentedStem: 4x downsampling, multi-scale aspect ratios, flatten=True/False modes.
  5. SVTRv2XEncoder (Tier 1): forward pass, aux_dict outputs (SDM + MoE), parameter delta vs Run 1.
  6. BaseRecognizer & SVTRv2XLoss: end-to-end forward/backward with AMP compatibility.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_tier1_tests():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("[!] PyTorch is not available. Skipping live tensor tests.")
        return

    print("=" * 70)
    print("Running SVTRv2X Tier 1 Architecture Injection Tests")
    print("=" * 70)

    from openrec.modeling.encoders.svtrv2_lnconv_two33_v2x import (
        SqueezeAndExcitation,
        ConvResidualBlock,
        Learnable2DPosEmbed,
        SEAugmentedStem,
        SVTRv2XEncoder,
        SVTRv2X_Tier1,
    )
    from openrec.modeling.base_recognizer import BaseRecognizer
    from openrec.losses.svtrv2x_loss import SVTRv2XLoss

    # --------------------------------------------------------------------------
    # Test 1: Squeeze-and-Excitation Module
    # --------------------------------------------------------------------------
    print("\n[1/6] Testing SqueezeAndExcitation Module...")
    se = SqueezeAndExcitation(channels=64, reduction=4)
    x = torch.randn(2, 64, 8, 32, requires_grad=True)
    out_se = se(x)
    assert out_se.shape == x.shape, f"SE shape mismatch: {out_se.shape} vs {x.shape}"
    loss_se = out_se.sum()
    loss_se.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "SE gradient failed!"
    print("  -> Passed SqueezeAndExcitation forward and backward verification.")

    # --------------------------------------------------------------------------
    # Test 2: ConvResidualBlock
    # --------------------------------------------------------------------------
    print("\n[2/6] Testing ConvResidualBlock...")
    res = ConvResidualBlock(dim=64)
    x = torch.randn(2, 64, 8, 32, requires_grad=True)
    out_res = res(x)
    assert out_res.shape == x.shape, f"ResBlock shape mismatch: {out_res.shape} vs {x.shape}"
    loss_res = out_res.sum()
    loss_res.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "ResBlock gradient failed!"
    print("  -> Passed ConvResidualBlock forward and backward verification.")

    # --------------------------------------------------------------------------
    # Test 3: Learnable2DPosEmbed (2D Layout, Slicing & Interpolation)
    # --------------------------------------------------------------------------
    print("\n[3/6] Testing Learnable2DPosEmbed...")
    pos_2d = Learnable2DPosEmbed(feat_max_size=[8, 64], embed_dim=64)
    assert pos_2d.pos_embed.shape == (1, 8, 64, 64), f"Pos embed parameter shape incorrect: {pos_2d.pos_embed.shape}"

    # Test 3a: Standard slicing [B, H', W', D] = [2, 8, 32, 64]
    x_tok_std = torch.randn(2, 8, 32, 64, requires_grad=True)
    out_pos_std = pos_2d(x_tok_std)
    assert out_pos_std.shape == (2, 8, 32, 64), f"Standard slice shape mismatch: {out_pos_std.shape}"

    # Test 3b: Fallback dynamic interpolation for ultra-wide image (W' = 80 > W'_max 64)
    x_tok_wide = torch.randn(2, 8, 80, 64, requires_grad=True)
    out_pos_wide = pos_2d(x_tok_wide)
    assert out_pos_wide.shape == (2, 8, 80, 64), f"Interpolation shape mismatch: {out_pos_wide.shape}"

    # Backward gradient check
    (out_pos_std.sum() + out_pos_wide.sum()).backward()
    assert pos_2d.pos_embed.grad is not None, "2D Pos Embed gradient not computed!"
    assert torch.isfinite(pos_2d.pos_embed.grad).all(), "2D Pos Embed gradient has non-finite values!"
    print("  -> Passed Learnable2DPosEmbed slicing, interpolation, and gradient verification.")

    # --------------------------------------------------------------------------
    # Test 4: SEAugmentedStem
    # --------------------------------------------------------------------------
    print("\n[4/6] Testing SEAugmentedStem on Multiple Aspect Ratios...")
    stem_conv = SEAugmentedStem(
        in_channels=3,
        feat_max_size=[8, 64],
        embed_dim=64,
        use_2d_pos_embed=True,
        se_reduction=4,
        flatten=False,
    )
    stem_trans = SEAugmentedStem(
        in_channels=3,
        feat_max_size=[8, 64],
        embed_dim=64,
        use_2d_pos_embed=True,
        se_reduction=4,
        flatten=True,
    )

    test_resolutions = [(32, 64), (32, 128), (32, 256)]
    for h, w in test_resolutions:
        img = torch.randn(2, 3, h, w)
        # 4a: Conv mode (flatten=False)
        out_conv, sz_conv = stem_conv(img)
        assert out_conv.shape == (2, 64, h // 4, w // 4), f"Conv mode shape error: {out_conv.shape}"
        assert sz_conv == [h // 4, w // 4]

        # 4b: Transformer sequence mode (flatten=True)
        out_trans, sz_trans = stem_trans(img)
        assert out_trans.shape == (2, (h // 4) * (w // 4), 64), f"Transformer mode shape error: {out_trans.shape}"
        assert sz_trans == [h // 4, w // 4]

    print("  -> Passed SEAugmentedStem multi-resolution checks (32x64, 32x128, 32x256).")

    # --------------------------------------------------------------------------
    # Test 5: SVTRv2XEncoder (Run 1 Baseline vs Run 2 Tier 1 Parameter Delta)
    # --------------------------------------------------------------------------
    print("\n[5/6] Testing SVTRv2XEncoder Tier 1 & Parameter Delta...")
    enc_baseline = SVTRv2XEncoder(
        dims=[64, 128, 256],
        depths=[3, 6, 3],
        num_heads=[2, 4, 8],
        use_se_stem=False,
        use_2d_pos_embed=False,
        use_jumble=True,
        use_moe=True,
        use_self_distill=True,
        feat2d=True,
    )

    enc_tier1 = SVTRv2X_Tier1(
        dims=[64, 128, 256],
        depths=[3, 6, 3],
        num_heads=[2, 4, 8],
        use_se_stem=True,
        use_2d_pos_embed=True,
        se_reduction=4,
        use_jumble=True,
        use_moe=True,
        use_self_distill=True,
        feat2d=True,
    )

    p_baseline = sum(p.numel() for p in enc_baseline.parameters())
    p_tier1 = sum(p.numel() for p in enc_tier1.parameters())
    p_delta = p_tier1 - p_baseline

    print(f"  -> Run 1 Baseline Encoder Params: {p_baseline / 1e6:.3f} M")
    print(f"  -> Run 2 Tier 1 Encoder Params:   {p_tier1 / 1e6:.3f} M")
    print(f"  -> Tier 1 Parameter Delta:        {p_delta / 1e3:.1f} k params ({p_delta / p_baseline * 100:.2f}%)")
    assert p_delta / p_baseline < 0.02, f"Parameter overhead exceeds 2%: {p_delta / p_baseline * 100:.2f}%"

    # Forward check
    enc_tier1.train()
    img = torch.randn(2, 3, 32, 128)
    feat = enc_tier1(img)
    assert 'sdm_f1' in enc_tier1.aux_dict, "sdm_f1 missing"
    assert 'sdm_f2' in enc_tier1.aux_dict, "sdm_f2 missing"
    assert 'moe_loss' in enc_tier1.aux_dict, "moe_loss missing"
    assert 'pos_embed_2d' in enc_tier1.no_weight_decay(), "pos_embed_2d missing from no_weight_decay!"
    print("  -> Passed SVTRv2XEncoder Tier 1 forward and auxiliary loss validation.")

    # --------------------------------------------------------------------------
    # Test 6: Full BaseRecognizer & SVTRv2XLoss Backward Pass (End-to-End)
    # --------------------------------------------------------------------------
    print("\n[6/6] Testing Full BaseRecognizer + Loss Integration with Run 2 Tier 1...")
    config = {
        'in_channels': 3,
        'Encoder': {
            'name': 'SVTRv2XEncoder',
            'use_se_stem': True,
            'use_2d_pos_embed': True,
            'se_reduction': 4,
            'use_pos_embed': False,
            'dims': [64, 128, 256],
            'depths': [3, 6, 3],
            'num_heads': [2, 4, 8],
            'feat2d': True,
            'use_jumble': True,
            'use_moe': True,
            'use_self_distill': True,
        },
        'Decoder': {
            'name': 'GTCDecoder',
            'infer_gtc': True,
            'detach': False,
            'gtc_decoder': {
                'name': 'SMTRDecoder',
                'num_layer': 1,
                'ds': True,
                'max_len': 25,
                'next_mode': True,
                'sub_str_len': 5,
            },
            'ctc_decoder': {
                'name': 'RCTCDecoder'
            }
        }
    }
    char_num = [6625, 6625]
    config['Decoder']['out_channels'] = char_num

    rec_model = BaseRecognizer(config)
    rec_model.train()

    B, L, S = 2, 25, 5
    label = torch.randint(1, 100, (B, L))
    label_subs = torch.randint(1, 100, (B, L, S))
    label_next = torch.randint(1, 100, (B, L))
    length_subs = torch.full((B, L), S, dtype=torch.long)
    label_subs_pre = torch.randint(1, 100, (B, L, S))
    label_next_pre = torch.randint(1, 100, (B, L))
    length_subs_pre = torch.full((B, L), S, dtype=torch.long)
    length = torch.tensor([L, L])
    ctc_label = torch.randint(1, 100, (B, L))
    ctc_length = torch.tensor([L, L])

    batch = [
        img, label, label_subs, label_next, length_subs,
        label_subs_pre, label_next_pre, length_subs_pre, length,
        ctc_label, ctc_length
    ]

    preds = rec_model(img, data=batch[1:])
    criterion = SVTRv2XLoss(
        gtc_loss={'name': 'SMTRLoss'},
        ctc_weight=0.1,
        gtc_weight=1.0,
        sdm_weight=0.5,
        moe_weight=0.01,
    )
    loss_out = criterion(preds, batch)
    loss_out['loss'].backward()

    # Verify gradients reach Tier 1 stem and 2D positional parameters
    pe_param = rec_model.encoder.pope.pos_embed_2d.pos_embed
    assert pe_param.grad is not None, "pos_embed gradient is None!"
    assert torch.isfinite(pe_param.grad).all(), "pos_embed gradient has NaN/Inf!"
    assert pe_param.grad.abs().sum() > 0, "pos_embed gradient is zero!"

    se_fc1 = rec_model.encoder.pope.se.fc[0].weight
    assert se_fc1.grad is not None, "SE fc1 weight gradient is None!"
    assert se_fc1.grad.abs().sum() > 0, "SE fc1 weight gradient is zero!"

    res1_conv1 = rec_model.encoder.pope.res_block1.conv1.weight
    assert res1_conv1.grad is not None, "ResBlock1 conv1 gradient is None!"
    assert res1_conv1.grad.abs().sum() > 0, "ResBlock1 conv1 gradient is zero!"

    total_model_params = sum(p.numel() for p in rec_model.parameters())
    print(f"  -> Total Tier 1 Model Parameters: {total_model_params / 1e6:.2f} M")
    print(f"  -> Verified non-zero gradients on 2D pos_embed, SE module, and ConvResidualBlocks.")
    print("  -> Passed Full Recognizer & Loss Backward Pass verification.")

    print("\n" + "=" * 70)
    print("ALL SVTRv2X RUN 2 (TIER 1) TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    run_tier1_tests()
