import torch
from torch import nn


class CTCLoss(nn.Module):

    def __init__(self, use_focal_loss=False, zero_infinity=True, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0,
                                    reduction='none',
                                    zero_infinity=zero_infinity)
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        batch_size = predicts.size(0)
        label, label_length = batch[1], batch[2]
        # Always run CTC log_softmax in fp32 to prevent half-precision underflow/overflow
        predicts = predicts.float().log_softmax(2)
        predicts = predicts.permute(1, 0, 2)
        preds_lengths = torch.tensor([predicts.size(0)] * batch_size,
                                     dtype=torch.long,
                                     device=predicts.device)
        loss = self.loss_func(predicts, label, preds_lengths, label_length)

        # Zero out any non-finite loss entries
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        if self.use_focal_loss:
            # Use torch.clamp to limit the range of loss, avoiding overflow in exponential calculation
            clamped_loss = torch.clamp(loss, min=-20, max=20)
            weight = 1 - torch.exp(-clamped_loss)
            weight = torch.square(weight)
            # Use torch.where to avoid multiplying by zero weight
            loss = torch.where(weight > 0, loss * weight, loss)
        loss = loss.mean()
        return {'loss': loss}
