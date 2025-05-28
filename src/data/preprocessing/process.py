from src.data.preprocessing.naver_processor import process_naver_dataset
from src.data.preprocessing.open_earth_map_processor import process_openearthmap_dataset

def process_dataset(config):
  if config.get('dataset_name', None) == 'open_earth_map':
    process_openearthmap_dataset(config)
  elif config.get('dataset_name', None) == 'naver':
    process_naver_dataset(config)
  else:
    raise ValueError(f"Dataset name {config.get('dataset_name', None)} not supported")
