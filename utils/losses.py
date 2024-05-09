import torch
from torch import nn


class CELabelSmoothedLoss(nn.Module):
    def __init__(self, n, alpha=0.0):
        super().__init__()

        self.alpha = alpha
        self.size = n

    def forward(self, y_pred, y_true):
        y_pred = y_pred.log_softmax(dim=-1)

        with torch.no_grad():
            target_probs = torch.zeros_like(y_pred)
            target_probs.fill_(self.alpha / (self.size - 1))
            target_probs.scatter_(1, y_true.data.unsqueeze(1), 1 - self.alpha)

        return torch.mean(torch.sum(-target_probs * y_pred, dim=-1))
