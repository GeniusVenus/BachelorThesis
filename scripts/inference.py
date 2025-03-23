import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.argument import parse_inference_args
from src.utils.config import load_config_from_inference_args
from src.models import SegmentationModel
from src.utils.checkpoint import load_checkpoint
from src.inference.predictor import Predictor

os.environ['CLEARML_CONFIG_FILE'] = '../clearml.conf'

def main():
    # Parse command line arguments
    args = parse_inference_args()

    print("==============Arguments==============")
    print(args)
    
    # Load and merge configurations
    config = load_config_from_inference_args(args)
    
    print("==============Configs==============")
    print(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SegmentationModel.get_model(config['model']['name'], **config['model'].get('params', {}))
    
    input_path = config['inference']['input_path']
    output_path = config['inference']['output_path']

    for checkpoint in config['inference']['checkpoints']:
        load_checkpoint(model, checkpoint)
    
        # Initialize predictor
        predictor = Predictor(model=model, config=config, device=device)
        
        predictor.predict_large_image(input_path, output_path)

if __name__ == '__main__':
    main()
