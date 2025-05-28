import albumentations as A
from albumentations.pytorch import ToTensorV2

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_transforms(config, split):
    if split == 'train':
        return A.Compose([
            # Spatial transformations - safe for both images and masks
            A.CenterCrop(256, 256, p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.GridDistortion(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),

            # Pixel-level transforms - only apply to images, not masks
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

            # Normalize and convert - apply to both but differently
            A.Normalize(mean=config['mean'], std=config['std'], max_pixel_value=255.0),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Normalize(mean=config['mean'], std=config['std'], max_pixel_value=255.0),
            ToTensorV2(),
        ])
