#!/usr/bin/env bash

# ==============================================================================
# SVTRv2X Training Scripts (Guo et al., PLOS ONE 2026)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. SVTRv2X Two-Phase Training Strategy (Section 3.3)
# ------------------------------------------------------------------------------

# Phase 1: Train SVTRv2X (JM + MoE) without Self-Distillation loss to establish stable representations
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2x_ch_phase1.yml

# Phase 2: Fine-tune for additional 20 epochs with Self-Distillation enabled (lambda_3 = 0.5)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2x_ch_phase2.yml

# Single-stage Joint Training (All modules active: JM + SDM + MoE)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_ch_jmsdmoe.yml


# ------------------------------------------------------------------------------
# 2. Ablation Studies (Table 2 in Paper)
# ------------------------------------------------------------------------------

# Row 1: Baseline (SVTRv2-Tiny)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_ch_base.yml

# Row 2: + JM (Jumble Module alone)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_ch_jm.yml

# Row 3: + SDM (Self-Distillation Module alone)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_ch_sd.yml

# Row 4: + MoE (Mixture of Experts alone)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_ch_moe.yml

# Row 5: + SDM + MoE
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_ch_jmsd.yml

# Row 6: + JM + SDM + MoE (Full SVTRv2X)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_ch_jmsdmoe.yml
