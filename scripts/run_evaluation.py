import sys
import os
import logging
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_evaluation_data
from src.search_engine import SemanticSearchEngine
from src.evaluation import evaluate_search_engine
from src import config
from src.utils import setup_logger, timeit

logger = logging.getLogger(__name__)

@timeit
def main():
    setup_logger()
    logger.info("--- Starting Search Engine Evaluation ---")

    # 1. Load Evaluation Data (Queries and Judgments)
    logger.info("Loading evaluation data...")
    evaluation_data = load_evaluation_data(locale='us') # Adjust locale as needed

    if not evaluation_data:
        logger.error("No evaluation data loaded. Exiting.")
        return

    #  Subset data for quicker testing
    # evaluation_data = evaluation_data[:100]
    # logger.warning(f"Using a subset of {len(evaluation_data)} queries for evaluation.")

    # 2. Initialize Search Engine
    logger.info("Initializing semantic search engine...")
    try:
        search_engine = SemanticSearchEngine()
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        logger.error("Ensure 'generate_embeddings.py' has been run successfully to create the index.")
        return

    if search_engine.vector_store.index is None:
        logger.error("Search engine initialized, but the vector index is missing or failed to load.")
        logger.error("Please run 'scripts/generate_embeddings.py' first.")
        return

    # 3. Run Evaluation
    logger.info(f"Running evaluation with K values: {config.EVALUATION_K}")
    metrics = evaluate_search_engine(search_engine, evaluation_data, k_values=config.EVALUATION_K)

    # 4. Print Results
    logger.info("--- Evaluation Complete ---")
    print("\n--- Evaluation Metrics ---")
    print(json.dumps(metrics, indent=2))
    print("------------------------")

    # 5. Save metrics to a file
    metrics_filename = f"evaluation_results_{config.EMBEDDING_MODEL_NAME.replace('/','_')}.json"
    results_path = os.path.join("..", metrics_filename) # Save in project root
    try:
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation metrics saved to {results_path}")
    except IOError as e:
        logger.error(f"Failed to save metrics to file: {e}")


if __name__ == "__main__":
    main() 