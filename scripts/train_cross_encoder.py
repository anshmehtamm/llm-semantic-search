# scripts/train_cross_encoder.py

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
import math
from collections import defaultdict # Helpful for grouping

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.utils import setup_logger, timeit

setup_logger()
logger = logging.getLogger(__name__)


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
        row.get('product_bullet_point')
    ]
    return " ".join(str(part) for part in text_parts if pd.notna(part) and str(part).strip())

@timeit
def load_and_prepare_split_data(locale: str = 'us', validation_split_ratio: float = config.CROSS_ENCODER_VALIDATION_SPLIT):
    """Loads, prepares, and splits data into train/validation DataFrames."""

    logger.info(f"Loading examples data from: {config.EXAMPLES_DATA_PATH}")
    try:
        examples_df = pd.read_parquet(config.EXAMPLES_DATA_PATH)
        logger.info(f"Loaded {len(examples_df)} total examples.")
    except FileNotFoundError:
        logger.error(f"Examples data file not found at {config.EXAMPLES_DATA_PATH}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading examples data: {e}")
        return None, None

    logger.info(f"Filtering examples for locale='{locale}' and split='train'")
    filtered_examples_df = examples_df[
        (examples_df['product_locale'] == locale) & (examples_df['split'] == 'train')
    ].copy()
    logger.info(f"Found {len(filtered_examples_df)} examples for training (locale='{locale}', split='train').")

    if filtered_examples_df.empty:
        logger.error("No training examples found after filtering.")
        return None, None

    filtered_examples_df['score'] = filtered_examples_df['esci_label'].map(LABEL_TO_SCORE)
    original_count = len(filtered_examples_df)
    filtered_examples_df.dropna(subset=['score'], inplace=True)
    if len(filtered_examples_df) < original_count:
        logger.warning(f"Dropped {original_count - len(filtered_examples_df)} rows due to missing score mapping.")

    logger.info(f"Loading product data from: {config.PRODUCT_DATA_PATH}")
    try:
        products_df = pd.read_parquet(config.PRODUCT_DATA_PATH)
        products_df = products_df[products_df['product_locale'] == locale]
        logger.info(f"Loaded {len(products_df)} products for locale '{locale}'.")
    except FileNotFoundError:
        logger.error(f"Product data file not found at {config.PRODUCT_DATA_PATH}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading product data: {e}")
        return None, None

    logger.info("Preparing product text from product details...")
    products_df['product_text'] = products_df.apply(prepare_text_for_product, axis=1)

    products_to_merge = products_df[['product_id', 'product_text']]
    examples_to_merge = filtered_examples_df[['query_id', 'query', 'product_id', 'score']]

    logger.info("Merging filtered examples with product text...")
    merged_df = pd.merge(
        examples_to_merge,
        products_to_merge,
        on='product_id',
        how='inner'
    )
    logger.info(f"Merged data contains {len(merged_df)} query-product pairs.")

    if merged_df.empty:
        logger.error("Data is empty after merging examples and products.")
        return None, None

    logger.info(f"Splitting the merged training data into actual train/validation ({validation_split_ratio * 100}% validation)...")
    if len(merged_df) < 2:
         logger.error("Not enough data to perform train/validation split.")
         return None, None

    try:

        train_df, validation_df = train_test_split(
            merged_df,
            test_size=validation_split_ratio,
            random_state=42,
           
        )
        logger.info(f"Train set size: {len(train_df)}, Validation set size: {len(validation_df)}")
        return train_df, validation_df
    except Exception as e:
        logger.error(f"Error during train/validation split: {e}")
        return None, None


@timeit
def main():
    logger.info("--- Starting Cross-Encoder Fine-Tuning (using sentence-transformers) ---")

    # 1. Load and Prepare Data (Train/Validation Split)
    train_df, validation_df = load_and_prepare_split_data(locale='us')

    if train_df is None or validation_df is None:
        logger.error("Failed to load or prepare data. Exiting.")
        return

    # 2. Create InputExamples for Training
    logger.info("Creating InputExamples for the training set...")
    train_examples = []
    for _, row in train_df.iterrows():
        train_examples.append(
            InputExample(texts=[row['query'], row['product_text']], label=float(row['score']))
        )
    logger.info(f"Created {len(train_examples)} InputExamples for training.")

    # 3. Load Base Cross-Encoder Model
    logger.info(f"Loading base cross-encoder model: {config.BASE_CROSS_ENCODER_MODEL}")
    model = CrossEncoder(
        config.BASE_CROSS_ENCODER_MODEL,
        num_labels=1, # Still training for regression score prediction
        device=config.DEVICE,
        max_length=512
    )

    # 4. Prepare Validation Samples for CrossEncoderRerankingEvaluator
    logger.info(f"Preparing validation samples for CrossEncoderRerankingEvaluator (Positive threshold >= {POSITIVE_THRESHOLD})...")
    validation_samples = []
    # Group validation data by query_id
    grouped_val = validation_df.groupby('query_id')

    skipped_queries_count = 0
    for qid, group in grouped_val:
        query_text = group['query'].iloc[0] # Get query text (should be same for all rows in group)
        positive_docs = []
        negative_docs = []

        for _, row in group.iterrows():
            if row['score'] >= POSITIVE_THRESHOLD:
                positive_docs.append(row['product_text'])
            else:
                negative_docs.append(row['product_text'])

        total_docs = len(positive_docs) + len(negative_docs)

        # Only add sample if there's at least one positive AND more than one document in total
        if positive_docs and total_docs > 1:
            validation_samples.append({
                'query': query_text,
                'positive': positive_docs,
                'negative': negative_docs
            })
        else:
            if not positive_docs:
                 logger.debug(f"Query ID {qid} has no positive examples in validation set, skipping for evaluator.")
            elif total_docs <= 1:
                 logger.debug(f"Query ID {qid} has only {total_docs} total doc(s) in validation set, skipping for evaluator.")
            skipped_queries_count += 1
            pass

    logger.info(f"Created {len(validation_samples)} samples for the reranking evaluator.")
    if skipped_queries_count > 0:
        logger.warning(f"Skipped {skipped_queries_count} queries during validation sample preparation due to having <= 1 total documents or no positive documents.")

    # Check if data is populated before creating evaluator
    if validation_samples:
         evaluator = CrossEncoderRerankingEvaluator(
             samples=validation_samples,
             name='validation',
             batch_size=config.CROSS_ENCODER_TRAIN_BATCH_SIZE * 2, 
             show_progress_bar=False 
         )
         logger.info(f"CrossEncoderRerankingEvaluator created. Primary metric: {evaluator.primary_metric}")
    else:
         logger.warning("Could not prepare validation samples (>=2 docs total, >=1 positive) for CrossEncoderRerankingEvaluator. Skipping evaluation during training.")
         evaluator = None

    

    # 5. Fine-tune the Model
    steps_per_epoch = math.ceil(len(train_examples) / config.CROSS_ENCODER_TRAIN_BATCH_SIZE)
    evaluation_steps = max(100, int(steps_per_epoch / 5))
    logger.info(f"Steps per epoch: ~{steps_per_epoch}. Evaluation steps: {evaluation_steps}")

    warmup_steps = min(config.CROSS_ENCODER_WARMUP_STEPS, int(steps_per_epoch * 0.1))
    logger.info(f"Warmup steps: {warmup_steps}")

    output_path = str(config.FINETUNED_CROSS_ENCODER_PATH)
    logger.info(f"Starting training for {config.CROSS_ENCODER_EPOCHS} epochs...")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Using batch size: {config.CROSS_ENCODER_TRAIN_BATCH_SIZE}")

    model.fit(
        train_dataloader=DataLoader(train_examples, shuffle=True, batch_size=config.CROSS_ENCODER_TRAIN_BATCH_SIZE),
        evaluator=evaluator,
        epochs=config.CROSS_ENCODER_EPOCHS,
        evaluation_steps=evaluation_steps if evaluator else 0,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': config.CROSS_ENCODER_LR},
        output_path=output_path,
        show_progress_bar=True,
        save_best_model=True if evaluator else False
    )

    if not evaluator and not os.path.exists(os.path.join(output_path, 'pytorch_model.bin')):
        logger.info(f"Saving final model (no validation performed during training) to {output_path}")
        model.save(output_path)
    elif evaluator:
         logger.info(f"Best model saved during training based on evaluator's primary metric ({evaluator.primary_metric}) to {output_path}")

    logger.info("--- Cross-Encoder Fine-Tuning Complete ---")

if __name__ == "__main__":
    main()