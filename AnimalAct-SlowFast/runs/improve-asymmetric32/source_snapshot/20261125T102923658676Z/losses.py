"""Multi-label loss variants; original mathematical implementation.

ASL reference: Ridnik et al., ICCV 2021, Asymmetric Loss for Multi-Label
Classification, https://openaccess.thecvf.com/content/ICCV2021/html/
Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html
This implementation averages label decisions rather than using the reference's sum.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_negative=4.0, gamma_positive=0.0, clip=.05):
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.clip = clip

    def forward(self, logits, targets):
        logits = logits.float()
        positive = logits.sigmoid()
        negative = (1 - positive + self.clip).clamp(max=1)
        positive_term = targets * F.logsigmoid(logits)
        negative_log = F.logsigmoid(-logits) if not self.clip else negative.clamp_min(1e-8).log()
        with torch.no_grad():
            correct_probability = targets * positive + (1 - targets) * negative
            exponent = targets * self.gamma_positive + (1 - targets) * self.gamma_negative
            weight = (1 - correct_probability).clamp_min(0).pow(exponent)
        return -((positive_term + (1 - targets) * negative_log) * weight).mean()
