import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.models import SegmentationModel
from src.losses import SegmentationLoss
from src.utils.config import load_config_from_evaluation_args
from src.utils.checkpoint import load_checkpoint
from src.evaluation.evaluator import Evaluator
from src.data.naver_dataset import NAVERDataset
from src.utils.argument import parse_evaluation_args
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
        task_type="monitor",
        tags=[config['loss']['name'], config['model']['name'], backbone, config['data']['name']],
    )

    # Connect configuration to the task
    task.connect_configuration(config)

    # Log the command line arguments
    task.connect(vars(args))

    return task

def main():
    # Parse command line arguments
    args = parse_evaluation_args()
    print("==============Arguments==============")
    print(args)
    
    # Load and merge configurations
    config = load_config_from_evaluation_args(args)
    
    print("\n==============Config==============")
    print(config)
    
    # Setup ClearML
    task = setup_clearml(args, config)

    # task = None

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SegmentationModel.get_model(config['model']['name'], **config['model'].get('params', {}))
    
    # Initialize loss function
    loss_params = config['loss'].get('params', {}) or {}  # Use empty dict if params is None
    criterion = SegmentationLoss.get_loss(config['loss']['name'], **loss_params)
    
    # Setup datasets and dataloaders
    train_dataset = NAVERDataset(config, split='train')
    val_dataset = NAVERDataset(config, split='val')
    # test_dataset = NAVERDataset(config, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    # Load checkpoint
    load_checkpoint(model, config['evaluation']['checkpoint'])
    
    # Initialize evaluator with ClearML task
    evaluator = Evaluator(model=model, config=config, device=device, task=task)
    
    # Evaluate on all splits
    print("\n==============Evaluation==============")
    results = evaluator.evaluate_all_splits(train_loader, val_loader, None, criterion)   

    # Print summary of results
    print("\n==============Summary==============")
    print(f"Train Loss: {results['train']['loss']:.4f}, IoU: {results['train']['iou']:.4f}, Dice: {results['train']['dice']:.4f}, Accuracy: {results['train']['accuracy']:.4f}")
    print(f"Val Loss: {results['val']['loss']:.4f}, IoU: {results['val']['iou']:.4f}, Dice: {results['val']['dice']:.4f}, Accuracy: {results['val']['accuracy']:.4f}")

if __name__ == '__main__':
    main()
