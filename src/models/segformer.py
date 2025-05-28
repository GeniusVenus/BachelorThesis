from transformers import SegformerForSemanticSegmentation
from torch import nn

class SegFormer(nn.Module):
    def __init__(self, pretrained_model: str = "nvidia/mit-b0", classes: int = 1):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model,
            num_labels=classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        return self.model(x)