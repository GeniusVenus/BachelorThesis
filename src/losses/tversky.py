import torch.nn as nn
import segmentation_models_pytorch as smp


class TverskyLoss(nn.Module):
    def __init__(self, mode='multiclass', alpha=0.5, beta=0.5, smooth=1.0):
        """
        Tversky Loss from Segmentation Models PyTorch

        Args:
            mode (str): 'multiclass', 'multilabel', or 'binary'
            alpha (float): Weight of false positives
            beta (float): Weight of false negatives
            smooth (float): Smoothing factor to prevent division by zero
        """
        super().__init__()
        self.loss = smp.losses.TverskyLoss(
            mode=mode,
            alpha=alpha,
            beta=beta,
            smooth=smooth
        )

    def forward(self, input, target):
        return self.loss(input, target)