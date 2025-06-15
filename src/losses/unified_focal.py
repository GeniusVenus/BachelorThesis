import torch.nn as nn
from src.losses.focal import FocalLoss
from src.losses.focal_tversky import FocalTverskyLoss

class UnifiedFocalLoss(nn.Module):
    def __init__(
        self,
        lambda_weight=0.5,
        mode='multiclass',
        focal_alpha=None,
        focal_gamma=2.0,
        tversky_alpha=0.5,
        tversky_beta=0.5,
        tversky_gamma=1.0,
        smooth=1.0,
        reduction='mean'
    ):
        super().__init__()
        
        self.lambda_weight = lambda_weight
        
        # Initialize Focal Loss
        self.focal_loss = FocalLoss(
            mode=mode,
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction=reduction
        )
        
        # Initialize Focal Tversky Loss
        self.focal_tversky_loss = FocalTverskyLoss(
            mode=mode,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            smooth=smooth
        )
    
    def forward(self, y_pred, y_true):
        # Calculate individual losses
        focal = self.focal_loss(y_pred, y_true)
        focal_tversky = self.focal_tversky_loss(y_pred, y_true)
        
        # Combine using lambda weight
        unified_loss = self.lambda_weight * focal + (1 - self.lambda_weight) * focal_tversky
        
        return unified_loss
