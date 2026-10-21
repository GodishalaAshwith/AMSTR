import copy
from importlib import import_module
from torch import nn

name_to_module = {
    'ABINetLoss': '.abinet_loss',
    'ARLoss': '.ar_loss',
    'CDistNetLoss': '.cdistnet_loss',
    'CELoss': '.ce_loss',
    'CPPDLoss': '.cppd_loss',
    'CTCLoss': '.ctc_loss',
    'IGTRLoss': '.igtr_loss',
    'LISTERLoss': '.lister_loss',
    'LPVLoss': '.lpv_loss',
    'MGPLoss': '.mgp_loss',
    'PARSeqLoss': '.parseq_loss',
    'RobustScannerLoss': '.robustscanner_loss',
    'SEEDLoss': '.seed_loss',
    'SMTRLoss': '.smtr_loss',
    'SRNLoss': '.srn_loss',
    'VisionLANLoss': '.visionlan_loss',
    'CAMLoss': '.cam_loss',
    'SVTRv2XLoss': '.svtrv2x_loss',
}


def build_loss(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')

    if module_name in globals():
        module_class = globals()[module_name]
    else:
        assert module_name in name_to_module, Exception(
            '{} is not supported. The losses in {} are supportes'.format(
                module_name, list(name_to_module.keys())))
        module_path = name_to_module[module_name]
        module = import_module(module_path, package=__package__)
        module_class = getattr(module, module_name)

    return module_class(**config)


class GTCLoss(nn.Module):

    def __init__(self,
                 gtc_loss,
                 gtc_weight=1.0,
                 ctc_weight=1.0,
                 sdm_weight=0.0,
                 moe_weight=0.0,
                 zero_infinity=True,
                 **kwargs):
        super(GTCLoss, self).__init__()
        # 动态构建CTCLoss
        ctc_config = {'name': 'CTCLoss', 'zero_infinity': zero_infinity}
        self.ctc_loss = build_loss(ctc_config)
        # 构建GTC损失
        self.gtc_loss = build_loss(gtc_loss)
        self.gtc_weight = gtc_weight
        self.ctc_weight = ctc_weight
        self.sdm_weight = sdm_weight
        self.moe_weight = moe_weight

    def forward(self, predicts, batch):
        import torch.nn.functional as F
        ctc_loss = self.ctc_loss(predicts['ctc_pred'],
                                 [None] + batch[-2:])['loss']
        gtc_loss = self.gtc_loss(predicts['gtc_pred'], batch[:-2])['loss']
        total_loss = self.ctc_weight * ctc_loss + self.gtc_weight * gtc_loss
        res = {
            'loss': total_loss,
            'ctc_loss': ctc_loss,
            'gtc_loss': gtc_loss
        }
        if self.sdm_weight > 0.0 and 'sdm_f1' in predicts and 'sdm_f2' in predicts:
            f1, f2 = predicts['sdm_f1'], predicts['sdm_f2']
            if f1.shape[2:] != f2.shape[2:]:
                f1 = F.adaptive_avg_pool2d(f1, f2.shape[2:])
            sdm_loss = F.mse_loss(f1, f2.detach())
            total_loss = total_loss + self.sdm_weight * sdm_loss
            res['sdm_loss'] = sdm_loss
            res['loss'] = total_loss
        if self.moe_weight > 0.0 and 'moe_loss' in predicts:
            moe_loss = predicts['moe_loss']
            if hasattr(moe_loss, 'requires_grad') and moe_loss.requires_grad:
                total_loss = total_loss + self.moe_weight * moe_loss
                res['moe_loss'] = moe_loss
                res['loss'] = total_loss
        return res
