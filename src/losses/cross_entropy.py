import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=-100, label_smoothing=0.0):
        """
        Cross Entropy Loss with customizable parameters

        Args:
            weight (Tensor, optional): Manual rescaling weight for each class
            reduction (str): Reduction method ('mean', 'sum', or 'none')
            ignore_index (int): Specifies a target value that is ignored
            label_smoothing (float): Amount of label smoothing to apply
        """
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

    def forward(self, input, target):
        return self.loss(input, target)