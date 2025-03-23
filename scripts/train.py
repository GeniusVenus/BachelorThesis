import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.losses import SegmentationLoss
from src.models import SegmentationModel
from src.trainers import Trainer
from src.utils.argument import parse_training_args
from src.utils.config import load_config_from_training_args
from src.data.naver_dataset import NAVERDataset
from src.utils.metric import AverageMeter

os.environ['CLEARML_CONFIG_FILE'] = '../clearml.conf'

def main():
    # Parse command line arguments
    args = parse_training_args()

    print("==============Arguments==============")
    print(args)

    # Load and merge configurations
    config = load_config_from_training_args(args)

    print("==============Configs==============")
    print(config)

    # Setup ClearML
    # task = setup_clearml(args, config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup datasets and dataloaders
    train_dataset = NAVERDataset(config, split='train')
    val_dataset = NAVERDataset(config, split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    # Initialize model
    model = SegmentationModel.get_model(config['model']['name'], **config['model'].get('params', {})).to(device)

    # Initialize loss function
    loss_params = config['loss'].get('params', {}) or {}  # Use empty dict if params is None
    loss_fn = SegmentationLoss.get_loss(config['loss']['name'], **loss_params)

    # Initialize metrics
    metric_meters = {
        'train_loss': AverageMeter(), 'train_dice': AverageMeter(),
        'train_iou': AverageMeter(), 'train_acc': AverageMeter(),
        'val_loss': AverageMeter(), 'val_dice': AverageMeter(),
        'val_iou': AverageMeter(), 'val_acc': AverageMeter()
    }

    # Initialize trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        metric_meters=metric_meters,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Resume from checkpoint if specified
    # if args.resume:
    #     trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
