import torch.nn as nn
import segmentation_models_pytorch as smp

class DiceLoss(nn.Module):
    def __init__(self, mode='multiclass', classes=None, log_loss=False, smooth=1.0):
        """
        Dice Loss from Segmentation Models PyTorch

        Args:
            mode (str): 'multiclass', 'multilabel', or 'binary'
            classes (list): List of classes to include in loss calculation
            log_loss (bool): Apply logarithmic scaling to the loss
            smooth (float): Smoothing factor to prevent division by zero
        """
        super().__init__()
        self.loss = smp.losses.DiceLoss(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            smooth=smooth
        )

    def forward(self, input, target):
        return self.loss(input, target)