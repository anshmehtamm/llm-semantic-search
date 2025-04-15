import sys
import os
import numpy as np
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_products
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore
from src import config
from src.utils import setup_logger, timeit

logger = logging.getLogger(__name__)

def prepare_text_for_embedding(row: pd.Series) -> str:
    """
    Combine relevant product fields into a single text for embedding.
    """
    # Combine title, description, and bullet points
    text_parts = [
        row['product_title'],
        row['product_description'],
        row['product_bullet_point'],
        f"Brand: {row['product_brand']}",
        f"Color: {row['product_color']}"
    ]
    
    # Filter out None/NaN 
    return " ".join(str(part) for part in text_parts if pd.notna(part))

@timeit
def main():
    setup_logger()
    logger.info("--- Starting Embedding Generation and Index Building ---")

    # 1. Load Product Data
    products_df = load_products(locale='us')
    logger.info(f"Loaded {len(products_df)} products")
    
    # Prepare text for embedding
    logger.info("Preparing text for embedding...")
    texts_to_embed = products_df.apply(prepare_text_for_embedding, axis=1).tolist()
    product_ids = products_df['product_id'].tolist()
    
    # Log sample of prepared text
    logger.info("Sample of prepared text for embedding:")
    logger.info(texts_to_embed[0][:200] + "...")

    # 2. Initialize Embedding Generator
    logger.info(f"Initializing embedding generator with model: {config.EMBEDDING_MODEL_NAME}")
    embedder = EmbeddingGenerator()

    # 3. Generate Embeddings
    logger.info("Generating embeddings for products...")
    embeddings = embedder.encode(texts_to_embed)

    if embeddings is None or len(embeddings) == 0:
        logger.error("Embedding generation failed. Exiting.")
        return

    if len(embeddings) != len(product_ids):
         logger.error(f"Mismatch after embedding: {len(embeddings)} embeddings vs {len(product_ids)} product IDs. Exiting.")
         return


    embeddings_np = np.array(embeddings)
    logger.info(f"Generated embeddings shape: {embeddings_np.shape}")
    logger.info("Initializing vector store...")
    vector_store = VectorStore(dimension=embedder.dimension)

    logger.info("Building vector store index...")
    vector_store.build_index(embeddings_np, product_ids)

    if vector_store.index is None:
        logger.error("Failed to build vector store index. Check logs for errors.")
    else:
        logger.info(f"Successfully built and saved index with {vector_store.index.ntotal} vectors.")
        logger.info(f"Index file: {config.INDEX_FILE}")
        logger.info(f"ID map file: {config.ID_MAP_FILE}")

    logger.info("--- Embedding Generation and Index Building Complete ---")

if __name__ == "__main__":
    main() 