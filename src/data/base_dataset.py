import os

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, config, split='train'):
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config = config
        self.split = split
        self.setup()

    def setup(self):
        """Initialize dataset-specific setup."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError