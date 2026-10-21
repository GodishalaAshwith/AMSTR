import torch
import torch.nn as nn
import torch.nn.functional as F


class SVTRv2XLoss(nn.Module):
    """
    Combined Loss function for SVTRv2X (Guo et al., PLOS ONE 2026, Section 3.5, Eq 14):
      L = lambda_1 * L_ctc + lambda_2 * L_sgm + lambda_3 * L_sd + alpha * L_load
    
    Weights from paper:
      lambda_1 (ctc_weight): 0.1
      lambda_2 (gtc_weight): 1.0 (supervising SMTR bidirectional decoder / SGM)
      lambda_3 (sdm_weight): 0.5 (enabled in Phase 2, 0.0 in Phase 1)
      alpha    (moe_weight): 0.01 (auxiliary load balancing loss)
    """

    def __init__(
        self,
        gtc_loss,
        ctc_weight=0.1,
        gtc_weight=1.0,
        sdm_weight=0.5,
        moe_weight=0.01,
        zero_infinity=True,
        **kwargs,
    ):
        super(SVTRv2XLoss, self).__init__()
        from openrec.losses import build_loss

        ctc_config = {'name': 'CTCLoss', 'zero_infinity': zero_infinity}
        self.ctc_loss = build_loss(ctc_config)
        self.gtc_loss = build_loss(gtc_loss)

        self.ctc_weight = ctc_weight
        self.gtc_weight = gtc_weight
        self.sdm_weight = sdm_weight
        self.moe_weight = moe_weight

    def forward(self, predicts, batch):
        # 1. CTC Loss
        ctc_loss = self.ctc_loss(predicts['ctc_pred'], [None] + batch[-2:])['loss']

        # 2. SGM / GTC Loss
        gtc_loss = self.gtc_loss(predicts['gtc_pred'], batch[:-2])['loss']

        total_loss = self.ctc_weight * ctc_loss + self.gtc_weight * gtc_loss

        loss_dict = {
            'ctc_loss': ctc_loss,
            'gtc_loss': gtc_loss,
        }

        # 3. Self-Distillation Loss: ||F'_1 - sg(F'_2)||^2
        if self.sdm_weight > 0.0 and 'sdm_f1' in predicts and 'sdm_f2' in predicts:
            f1 = predicts['sdm_f1']
            f2 = predicts['sdm_f2']
            if f1.shape[2:] != f2.shape[2:]:
                f1 = F.adaptive_avg_pool2d(f1, f2.shape[2:])
            # Teacher features f2 are gradient-stopped
            sdm_loss = F.mse_loss(f1, f2.detach())
            total_loss = total_loss + self.sdm_weight * sdm_loss
            loss_dict['sdm_loss'] = sdm_loss

        # 4. MoE Auxiliary Load Balancing Loss
        if self.moe_weight > 0.0 and 'moe_loss' in predicts:
            moe_loss = predicts['moe_loss']
            if isinstance(moe_loss, torch.Tensor) and moe_loss.requires_grad:
                total_loss = total_loss + self.moe_weight * moe_loss
                loss_dict['moe_loss'] = moe_loss

        loss_dict['loss'] = total_loss
        return loss_dict
