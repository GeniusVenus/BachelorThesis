import torch.nn as nn
import segmentation_models_pytorch as smp

class FocalLoss(nn.Module):
    def __init__(self, mode='multiclass', alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss from Segmentation Models PyTorch

        Args:
            mode (str): 'multiclass', 'multilabel', or 'binary'
            alpha (float or list): Weighting factor for each class
            gamma (float): Focusing parameter
            reduction (str): Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.loss = smp.losses.FocalLoss(
            mode=mode,
            alpha=alpha,
            gamma=gamma,
            reduction=reduction
        )

    def forward(self, input, target):
        return self.loss(input, target)