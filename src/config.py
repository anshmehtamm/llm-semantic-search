import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / 'data'

# Dataset files
PRODUCT_DATA_PATH = DATA_DIR / 'shopping_queries_dataset_products.parquet'
EXAMPLES_DATA_PATH = DATA_DIR / 'shopping_queries_dataset_examples.parquet'
SOURCES_DATA_PATH = DATA_DIR / 'shopping_queries_dataset_sources.csv'

# Index directory
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = PROJECT_ROOT / 'index'
INDEX_FILE = INDEX_DIR / f'{EMBEDDING_MODEL_NAME.replace("/","_")}_faiss.index'
ID_MAP_FILE = INDEX_DIR / f'{EMBEDDING_MODEL_NAME.replace("/","_")}_id_map.json'

# Model configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Search configuration
DEFAULT_SEARCH_K = 10
EVALUATION_K = [1, 3, 5, 10]

INDEX_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

EMBEDDING_BATCH_SIZE = 64

# Cross-Encoder Configuration
BASE_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FINETUNED_CROSS_ENCODER_PATH = PROJECT_ROOT / 'models' / 'fine_tuned_cross_encoder'
CROSS_ENCODER_TRAIN_BATCH_SIZE = 64
CROSS_ENCODER_EPOCHS = 1 
CROSS_ENCODER_LR = 2e-5
CROSS_ENCODER_WARMUP_STEPS = 100
CROSS_ENCODER_VALIDATION_SPLIT = 0.1

# Create model directory
FINETUNED_CROSS_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)

RERANK_CANDIDATE_COUNT = 50 # candidates to fetch from semantic search for re-ranking
