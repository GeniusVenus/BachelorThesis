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

    input_path = config['inference']['input']
    output_path = config['inference']['output']

    checkpoint = config['inference']['checkpoint']
    load_checkpoint(model, checkpoint)

    if task:
        # Log the checkpoint being used
        task.logger.report_text(f"Using checkpoint: {checkpoint}")

    # Initialize predictor
    predictor = Predictor(model=model, config=config, device=device, task=task)

    with open(input_path, 'r') as f:
        files = f.readlines()
        for file_name in files:
            # Log the input image being processed
            file = f"{config['data']['raw_paths']['image']}/{file_name}"
            if file_name.__contains__("patch"):
                file = f"{config['data']['dataset_path']}/test/images/{file_name}"
            if task:
                task.logger.report_text(f"Processing input: {file}")

            # Perform inference
            predictor.predict_large_image(file, output_path)

if __name__ == '__main__':
    main()
