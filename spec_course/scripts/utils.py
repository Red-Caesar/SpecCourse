import logging
import yaml
from pathlib import Path

def setup_logger(output_dir: str, log_name: str = "test") -> logging.Logger:
    log_file = Path(output_dir) / f"{log_name}.log"
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)