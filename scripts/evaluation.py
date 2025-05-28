import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

from src.models import SegmentationModel
from src.losses import SegmentationLoss
from src.utils.config import load_config_from_evaluation_args
from src.utils.checkpoint import load_checkpoint
from src.evaluation.evaluator import Evaluator
from src.data.naver_dataset import NAVERDataset
from src.utils.argument import parse_evaluation_args

os.environ['CLEARML_CONFIG_FILE'] = '../clearml.conf'

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
    test_dataset = NAVERDataset(config, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    # Load checkpoint
    for checkpoint in config['evaluation']['checkpoints']:
        load_checkpoint(model, checkpoint)
        # Initialize evaluator
        evaluator = Evaluator(model=model, config=config, device=device)
        
        # Evaluate on all splits
        print("\n==============Evaluation==============")
        results = evaluator.evaluate_all_splits(train_loader, val_loader, test_loader, criterion)
    
        # Print summary of results
        print("\n==============Summary==============")
        print(f"Train IoU: {results['train']['iou']:.4f}, Dice: {results['train']['dice']:.4f}")
        print(f"Val IoU: {results['val']['iou']:.4f}, Dice: {results['val']['dice']:.4f}")
        print(f"Test IoU: {results['test']['iou']:.4f}, Dice: {results['test']['dice']:.4f}")

if __name__ == '__main__':
    main()
