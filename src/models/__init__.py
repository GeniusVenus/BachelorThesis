from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.fpn import FPN
from src.models.pspnet import PSPNet
from src.models.segformer import SegFormer
from src.models.unet import UNet
from src.models.upernet import UperNet


class SegmentationModel:
    @staticmethod
    def get_model(model_name: str, **kwargs):
        models = {
            'unet': UNet,
            'deeplabv3plus': DeepLabV3Plus,
            'fpn': FPN,
            'pspnet': PSPNet,
            'segformer': SegFormer,
            'upernet': UperNet
        }

        if model_name.lower() not in models:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(models.keys())}")

        return models[model_name.lower()](**kwargs)