import segmentation_models_pytorch as smp
from torch import nn

class UNet(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", in_channels: int = 3,
                 classes: int = 1, encoder_weights: str = "imagenet"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )

    def forward(self, x):
        return self.model(x)