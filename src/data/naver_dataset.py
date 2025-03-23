import os
import cv2
from .base_dataset import BaseDataset
from .preprocessing.transforms import get_transforms

class NAVERDataset(BaseDataset):
    def __init__(self, config, split='train'):
        super().__init__(config, split)
        self.transforms = get_transforms(config['augmentation'], split)

    def setup(self):
        self.image_dir = os.path.join(self.project_root, self.config['data']['dataset_path'], self.split ,'images')
        self.mask_dir = os.path.join(self.project_root, self.config['data']['dataset_path'], self.split ,'labels')
        self.image_files = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_name = image_name.replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Read image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if image is None:
            raise ValueError(f"Could not read image {image_path}")

        if mask is None:
            raise ValueError(f"Could not load mask at {mask_path}")

        # Apply transforms
        transformed = self.transforms(image=image, mask=mask)

        return {
            'image': transformed['image'],
            'mask': transformed['mask'],
            'image_name': image_name
        }