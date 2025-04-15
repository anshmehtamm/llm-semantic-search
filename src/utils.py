import logging
import sys
import time
import json

def setup_logger():
    """Sets up the root logger."""
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)
    # Reduce verbosity of HuggingFace libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

def timeit(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger = logging.getLogger(func.__module__)
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def save_json(data, file_path):
    """Saves data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Successfully saved JSON data to {file_path}")
    except IOError as e:
        logging.error(f"Error saving JSON to {file_path}: {e}")

def load_json(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded JSON data from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except IOError as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return None

# Setup logger on import
setup_logger() 