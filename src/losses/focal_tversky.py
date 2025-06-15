import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class FocalTverskyLoss(nn.Module):
    def __init__(self, mode='multiclass', alpha=0.5, beta=0.5, gamma=1.0, smooth=1.0):
        """
        Focal Tversky Loss

        Args:
            mode (str): 'multiclass', 'multilabel', or 'binary'
            alpha (float): Weight of false positives in Tversky index
            beta (float): Weight of false negatives in Tversky index
            gamma (float): Focusing parameter for Focal component
            smooth (float): Smoothing factor to prevent division by zero
        """
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
        # Initialize underlying Tversky Loss
        self.tversky_loss = smp.losses.TverskyLoss(
            mode=mode,
            alpha=alpha,
            beta=beta,
            smooth=smooth
        )
    
    def forward(self, y_pred, y_true):
        tversky_loss = self.tversky_loss(y_pred, y_true)
        
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        
        return focal_tversky
