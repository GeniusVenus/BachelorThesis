import torch.nn as nn
import segmentation_models_pytorch as smp

class LovaszLoss(nn.Module):
    def __init__(self, mode='multiclass', per_image=False):
        """
        Lovasz Loss for semantic segmentation

        Args:
            mode (str): 'multiclass', 'multilabel', or 'binary'
            per_image (bool): Whether to compute the loss per image instead of per batch
        """
        super().__init__()
        self.loss = smp.losses.LovaszLoss(
            mode=mode,
            per_image=per_image
        )

    def forward(self, input, target):
        return self.loss(input, target)