# scripts/evaluate_reranker.py

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.utils import setup_logger, timeit

# --- Configuration (reuse from training script) ---
setup_logger()
logger = logging.getLogger(__name__)

# Define relevance thresholds for evaluation
POSITIVE_THRESHOLD = 0.75 

LABEL_TO_SCORE = {
    'E': 1.0,
    'S': 0.75,
    'C': 0.5,
    'I': 0.0
}

def prepare_text_for_product(row: pd.Series) -> str:
    """Combines relevant product fields into a single text string."""
    text_parts = [
        row.get('product_title'),
        f"Brand: {row.get('product_brand')}" if pd.notna(row.get('product_brand')) else None,
        f"Color: {row.get('product_color')}" if pd.notna(row.get('product_color')) else None,
        row.get('product_description'),
        row.get('product_bullet_point'),
    ]
    return " ".join(str(part) for part in text_parts if pd.notna(part) and str(part).strip())
# --- End Configuration ---


@timeit
def load_and_prepare_test_data(locale: str = 'us'):
    """
    Loads example and product data, filters by locale and 'test' split,
    merges them, prepares product text, and assigns scores.

    Returns:
        pd.DataFrame: Merged and prepared test data, or None if an error occurs.
    """
    logger.info(f"Loading examples data for evaluation from: {config.EXAMPLES_DATA_PATH}")
    try:
        examples_df = pd.read_parquet(config.EXAMPLES_DATA_PATH)
        logger.info(f"Loaded {len(examples_df)} total examples.")
    except FileNotFoundError:
        logger.error(f"Examples data file not found at {config.EXAMPLES_DATA_PATH}")
        return None
    except Exception as e:
        logger.error(f"Error loading examples data: {e}")
        return None

    # Filter by locale and split='test'
    logger.info(f"Filtering examples for locale='{locale}' and split='test'")
    test_examples_df = examples_df[
        (examples_df['product_locale'] == locale) & (examples_df['split'] == 'test')
    ].copy()
    logger.info(f"Found {len(test_examples_df)} examples for evaluation (locale='{locale}', split='test').")

    if test_examples_df.empty:
        logger.error("No test examples found after filtering. Check locale and split name.")
        return None

    # Assign numerical scores (needed for positive/negative split)
    test_examples_df['score'] = test_examples_df['esci_label'].map(LABEL_TO_SCORE)
    original_count = len(test_examples_df)
    test_examples_df.dropna(subset=['score'], inplace=True)
    if len(test_examples_df) < original_count:
        logger.warning(f"Dropped {original_count - len(test_examples_df)} test rows due to missing score mapping.")

    logger.info(f"Loading product data from: {config.PRODUCT_DATA_PATH}")
    try:
        products_df = pd.read_parquet(config.PRODUCT_DATA_PATH)
        products_df = products_df[products_df['product_locale'] == locale]
        logger.info(f"Loaded {len(products_df)} products for locale '{locale}'.")
    except FileNotFoundError:
        logger.error(f"Product data file not found at {config.PRODUCT_DATA_PATH}")
        return None
    except Exception as e:
        logger.error(f"Error loading product data: {e}")
        return None

    logger.info("Preparing product text from product details...")
    products_df['product_text'] = products_df.apply(prepare_text_for_product, axis=1)

    products_to_merge = products_df[['product_id', 'product_text']]
    examples_to_merge = test_examples_df[['query_id', 'query', 'product_id', 'score']]

    logger.info("Merging test examples with product text...")
    merged_test_df = pd.merge(
        examples_to_merge,
        products_to_merge,
        on='product_id',
        how='inner'
    )
    logger.info(f"Merged test data contains {len(merged_test_df)} query-product pairs.")

    if merged_test_df.empty:
        logger.error("Test data is empty after merging examples and products.")
        return None

    return merged_test_df


def prepare_evaluation_samples(test_df: pd.DataFrame):
    """
    Prepares the test data into the list of sample dictionaries required by
    CrossEncoderRerankingEvaluator.
    """
    logger.info(f"Preparing evaluation samples (Positive threshold >= {POSITIVE_THRESHOLD})...")
    evaluation_samples = []
    grouped_test = test_df.groupby('query_id')

    for qid, group in grouped_test:
        query_text = group['query'].iloc[0]
        positive_docs = []
        negative_docs = []

        for _, row in group.iterrows():
            if row['score'] >= POSITIVE_THRESHOLD:
                positive_docs.append(row['product_text'])
            else:
                negative_docs.append(row['product_text'])

        if positive_docs:
            evaluation_samples.append({
                'query': query_text,
                'positive': positive_docs,
                'negative': negative_docs
            })
        else:
             logger.debug(f"Query ID {qid} has no positive examples in test set, skipping for evaluator.")

    logger.info(f"Created {len(evaluation_samples)} samples for the reranking evaluator.")
    return evaluation_samples


@timeit
def main(args):
    logger.info(f"--- Starting Cross-Encoder Re-ranking Evaluation on Test Set ---")
    logger.info(f"Evaluating model: {args.model_path_or_name}")
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Evaluation locale: {args.locale}")

    # 1. Load and Prepare Test Data
    test_df = load_and_prepare_test_data(locale=args.locale)
    if test_df is None:
        logger.error("Failed to load or prepare test data. Exiting.")
        return

    # 2. Prepare Samples for Evaluator
    evaluation_samples = prepare_evaluation_samples(test_df)
    if not evaluation_samples:
        logger.error("No evaluation samples could be created from the test data (perhaps no queries with positive examples?). Exiting.")
        return

    # 3. Initialize Evaluator
    evaluator = CrossEncoderRerankingEvaluator(
        samples=evaluation_samples,
        name=f'test_set_{args.locale}',
        batch_size=args.batch_size,
        show_progress_bar=True 
    )
    logger.info(f"CrossEncoderRerankingEvaluator initialized. Expecting metrics like: {evaluator.primary_metric}")

    # 4. Load Cross-Encoder Model
    logger.info(f"Loading CrossEncoder model from: {args.model_path_or_name}")
    try:
        model = CrossEncoder(args.model_path_or_name, device=config.DEVICE, num_labels=1) # Assuming regression head
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Attempt loading without num_labels if it was saved differently (less likely for CrossEncoder)
        try:
            logger.warning("Retrying model load without specifying num_labels...")
            model = CrossEncoder(args.model_path_or_name, device=config.DEVICE)
        except Exception as e2:
            logger.error(f"Failed to load model even without num_labels: {e2}")
            return

    # 5. Run Evaluation
    logger.info("Running evaluation...")
    results = evaluator(model, output_path=args.output_dir)

    # 6. Log Results
    logger.info("--- Evaluation Results ---")
    if results:
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
    else:
        logger.warning("Evaluation did not return any results.")

    logger.info(f"CSV results potentially saved in: {args.output_dir or '.'}")
    logger.info("--- Evaluation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a CrossEncoder model for re-ranking on the test set.")
    parser.add_argument(
        "model_path_or_name",
        type=str,
        help="Path to the fine-tuned model directory OR Hugging Face model name (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2')."
    )
    parser.add_argument(
        "--locale",
        type=str,
        default="us",
        help="Locale to evaluate (e.g., 'us', 'es', 'jp')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None, 
        help="Directory to save the evaluation results CSV file."
    )

    args = parser.parse_args()

    # Use fine-tuned path from config if 'finetuned' is passed as model name
    if args.model_path_or_name.lower() == 'finetuned':
        args.model_path_or_name = str(config.FINETUNED_CROSS_ENCODER_PATH)
        if not os.path.isdir(args.model_path_or_name):
             logger.error(f"Specified 'finetuned' but path does not exist: {args.model_path_or_name}")
             sys.exit(1)

    # Ensure output directory exists if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)


    main(args)