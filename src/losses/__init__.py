from src.losses.cross_entropy import CrossEntropyLoss
from src.losses.dice import DiceLoss
from src.losses.focal import FocalLoss
from src.losses.jaccard import JaccardLoss
from src.losses.tversky import TverskyLoss
from src.losses.lovasz import LovaszLoss


class SegmentationLoss:
    @staticmethod
    def get_loss(loss_name: str, **kwargs):
        losses = {
            'cross_entropy': CrossEntropyLoss,
            'dice': DiceLoss,
            'focal': FocalLoss,
            'jaccard': JaccardLoss,
            'tversky': TverskyLoss,
            'lovasz': LovaszLoss,
        }
        
        if(loss_name == 'all'):
            return [losses[loss] for loss in losses.keys()]

        if loss_name.lower() not in losses:
            raise ValueError(f"Loss {loss_name} not available. Choose from: {list(losses.keys())}")

        return losses[loss_name.lower()](**kwargs)
