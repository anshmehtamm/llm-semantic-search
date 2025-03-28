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

# Create necessary directories
INDEX_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# --- Model Configuration ---
# Recommended starting model (good balance of performance and size)
# Other options: 'all-mpnet-base-v2', 'distilbert-base-nli-stsb-mean-tokens'
# Automatically detect dimension based on model, or set manually if needed
# EMBEDDING_DIMENSION = 384 # for all-MiniLM-L6-v2

# --- Data Configuration ---
# !!! IMPORTANT: Update this path to where you download the dataset !!!
# You might need separate paths for products, queries, and judgments
# PRODUCT_DATA_PATH = os.path.join(DATA_DIR, "products.csv") # Adjust filename/format
# QUERY_DATA_PATH = os.path.join(DATA_DIR, "queries.csv") # Adjust filename/format
# JUDGEMENT_DATA_PATH = os.path.join(DATA_DIR, "judgments.csv") # Adjust filename/format

# --- Index Configuration ---
# os.makedirs(INDEX_DIR, exist_ok=True) # Ensure index directory exists

# --- Search Configuration ---
# os.makedirs(INDEX_DIR, exist_ok=True) # Ensure index directory exists

# --- Evaluation Configuration ---
# os.makedirs(INDEX_DIR, exist_ok=True) # Ensure index directory exists

# --- Performance ---
EMBEDDING_BATCH_SIZE = 2*2048 # Adjust based on GPU memory

# --- LangChain ---
# Optional: Add API keys if using external services via LangChain
# OPENAI_API_KEY = "your_key_here" 