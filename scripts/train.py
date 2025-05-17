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

from clearml import Task

# os.environ['CLEARML_CONFIG_FILE'] = '../clearml.conf'

def setup_clearml(args, config):
    """Setup ClearML task for experiment tracking"""
    # Create a task
    if 'encoder_name' in config['model']['params']:
        backbone = config['model']['params']['encoder_name']
        task_name = f"{config['loss']['name']}_{config['model']['params']['encoder_name']}_{config['model']['name']}"
    else:
        backbone = config['model']['params']['pretrained_model']
        task_name = f"{config['loss']['name']}_{config['model']['params']['pretrained_model']}_{config['model']['name']}"

    project_name = config.get('clearml', {}).get('project_name', 'Segmentation')

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        tags=[config['loss']['name'], config['model']['name'], backbone, config['data']['name']],
    )

    # Connect configuration to the task
    task.connect_configuration(config)

    # Log the command line arguments
    task.connect(vars(args))

    return task

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
    task = setup_clearml(args, config)
    # task = None

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
        task=task,  # Pass the ClearML task to the trainer
    )

    # Resume from checkpoint if specified
    # if args.resume:
    #     trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
