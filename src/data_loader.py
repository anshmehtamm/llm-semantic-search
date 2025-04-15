import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import config

logger = logging.getLogger(__name__)

def load_products(locale: str = 'us') -> pd.DataFrame:
    """
    Load product data from the ESCI dataset parquet file.
    
    Args:
        locale: Locale code (e.g., 'us', 'es', 'jp')
        
    Returns:
        DataFrame containing product information
    """
    try:
        data_dir = Path(__file__).parent.parent / 'data'
        product_file = data_dir / 'shopping_queries_dataset_products.parquet'
        
        if not product_file.exists():
            raise FileNotFoundError(f"Product data file not found at {product_file}")
            
        # Read parquet file
        df = pd.read_parquet(product_file)
        
        # Filter by locale if specified
        if locale:
            df = df[df['product_locale'] == locale]
            
        # Rename columns to match our expected format
        df = df.rename(columns={
            'product_id': 'product_id',
            'product_title': 'product_title',
            'product_description': 'product_description',
            'product_bullet_point': 'product_bullet_point',
            'product_brand': 'product_brand',
            'product_color': 'product_color',
            'product_locale': 'product_locale'
        })
        
        logger.info(f"Loaded {len(df)} products from {product_file}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading product data: {str(e)}")
        raise

def load_evaluation_data(locale: str = 'us') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load evaluation queries and judgments from the ESCI dataset.
    
    Args:
        locale: Locale code (e.g., 'us', 'es', 'jp')
        
    Returns:
        Tuple of (queries DataFrame, judgments DataFrame)
    """
    try:
        data_dir = Path(__file__).parent.parent / 'data'
        examples_file = data_dir / 'shopping_queries_dataset_examples.parquet'
        
        if not examples_file.exists():
            raise FileNotFoundError(f"Examples data file not found at {examples_file}")
            
        # Read parquet file
        df = pd.read_parquet(examples_file)
        
        # Filter by locale if specified
        if locale:
            df = df[df['query_locale'] == locale]
            
        # Split into queries and judgments
        queries = df[['query_id', 'query', 'query_locale']].drop_duplicates()
        judgments = df[['query_id', 'product_id', 'esci_label']].rename(columns={'esci_label': 'relevance'})
        
        # Convert ESCI labels to relevance scores (1-4)
        label_to_score = {
            'E': 4,  # Exact
            'S': 3,  # Substitute
            'C': 2,  # Complement
            'I': 1   # Irrelevant
        }
        judgments['relevance'] = judgments['relevance'].map(label_to_score)
        
        logger.info(f"Loaded {len(queries)} queries and {len(judgments)} judgments")
        return queries, judgments
        
    except Exception as e:
        logger.error(f"Error loading evaluation data: {str(e)}")
        raise

def load_training_data(locale: str = 'us') -> pd.DataFrame:
    """
    Load training data from the ESCI dataset.
    
    Args:
        locale: Locale code (e.g., 'us', 'es', 'jp')
        
    Returns:
        DataFrame containing training examples
    """
    try:
        data_dir = Path(__file__).parent.parent / 'data'
        examples_file = data_dir / 'shopping_queries_dataset_examples.parquet'
        
        if not examples_file.exists():
            raise FileNotFoundError(f"Examples data file not found at {examples_file}")
            
        # Read parquet file
        df = pd.read_parquet(examples_file)
        
        # Filter by locale if specified
        if locale:
            df = df[df['query_locale'] == locale]
            
        # Convert ESCI labels to relevance scores
        label_to_score = {
            'E': 4,  # Exact
            'S': 3,  # Substitute
            'C': 2,  # Complement
            'I': 1   # Irrelevant
        }
        df['relevance'] = df['esci_label'].map(label_to_score)
        
        logger.info(f"Loaded {len(df)} training examples")
        return df
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

# testing
if __name__ == '__main__':
    logger.info("Testing data loaders...")
    try:
        dummy_prod_path = os.path.join(config.DATA_DIR, "products.csv")
        if not os.path.exists(dummy_prod_path):
             pd.DataFrame({
                'product_id': ['p1', 'p2', 'p3', 'p4'],
                'product_title': ['Running Shoes', 'Wireless Mouse', 'Coffee Maker', 'Libro Español'],
                'product_description': ['Comfortable running shoes', 'Ergonomic wireless mouse', 'Makes great coffee', 'Un libro sobre IA'],
                'product_locale': ['us', 'us', 'us', 'es']
            }).to_csv(dummy_prod_path, index=False)

        dummy_query_path = os.path.join(config.DATA_DIR, "queries.csv")
        if not os.path.exists(dummy_query_path):
            pd.DataFrame({
                'query_id': ['q1', 'q2', 'q3'],
                'query': ['best running shoes', 'mouse for gaming', 'libro de IA'],
                'query_locale': ['us', 'us', 'es']
            }).to_csv(dummy_query_path, index=False)

        dummy_judge_path = os.path.join(config.DATA_DIR, "judgments.csv")
        if not os.path.exists(dummy_judge_path):
            pd.DataFrame({
                'query_id': ['q1', 'q1', 'q2', 'q2', 'q3'],
                'product_id': ['p1', 'p2', 'p2', 'p4', 'p4'],
                'relevance_level': ['Exact', 'Irrelevant', 'Substitute', 'Irrelevant', 'Exact']
            }).to_csv(dummy_judge_path, index=False)

    except Exception as e:
        logger.error(f"Could not create dummy data files: {e}")


    products_us = load_products(locale='us')
    print(f"\nLoaded US Products:\n{products_us[:2]}")

    eval_data_us = load_evaluation_data(locale='us')
    print(f"\nLoaded US Evaluation Data:\n{eval_data_us[:2]}") 