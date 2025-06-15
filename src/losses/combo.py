import torch.nn as nn
import segmentation_models_pytorch as smp

class ComboLoss(nn.Module):
    def __init__(
        self, 
        alpha=0.5, 
        mode='multiclass', 
        classes=None, 
        smooth=1.0,
        ce_weight=None,
        ignore_index=-100,
        reduction='mean',
        label_smoothing=0.0
    ):
        super().__init__()
        
        self.alpha = alpha
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=ce_weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
        self.dice_loss = smp.losses.DiceLoss(
            mode=mode,
            classes=classes,
            log_loss=False,
            smooth=smooth,
            ignore_index=ignore_index
        )
        
    def forward(self, y_pred, y_true):
        """
        Calculate combined loss
        
        Args:
            y_pred (torch.Tensor): Model predictions (logits)
            y_true (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined loss value
        """
        ce_loss = self.ce_loss(y_pred, y_true)
        dice_loss = self.dice_loss(y_pred, y_true)
        
        combo_loss = self.alpha * ce_loss + (1 - self.alpha) * dice_loss
        
        return combo_loss
