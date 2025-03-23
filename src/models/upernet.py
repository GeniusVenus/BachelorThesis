from transformers import UperNetForSemanticSegmentation
from torch import nn

class UperNet(nn.Module):
    def __init__(self, pretrained_model: str = "openmmlab/upernet-convnext-tiny", classes: int = 1):
        super().__init__()
        self.model = UperNetForSemanticSegmentation.from_pretrained(
            pretrained_model,
            num_labels=classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        return self.model(x).logits