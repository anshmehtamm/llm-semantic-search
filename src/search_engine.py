import sys
import os
import logging
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

try:
    from . import config
except ImportError:
    import config

from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .utils import timeit, setup_logger
from .data_loader import load_products 


try:
    scripts_path = Path(__file__).parent.parent / 'scripts'
    sys.path.insert(0, str(scripts_path))
    from train_cross_encoder import prepare_text_for_product
    sys.path.pop(0) 
except ImportError:
    # Fallback or define it here if needed, but consistency is key
    logging.error("Could not import 'prepare_text_for_product' from training script.")

# Import CrossEncoder
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None
    logging.warning("sentence-transformers not installed. CrossEncoder re-ranking will not be available.")

# Setup logger
setup_logger()
logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """Orchestrates embedding generation, vector search, and optional re-ranking."""
    def __init__(self, locale: str = 'us', model_type: str = 'finetuned'):
        """
        Initializes the search engine.

        Args:
            locale (str): The data locale.
            model_type (str): 'finetuned' to attempt loading the fine-tuned reranker,
                              'base' to load the base reranker model.
        """
        logger.info(f"Initializing Semantic Search Engine for locale '{locale}'...")
        self.embedder = EmbeddingGenerator(model_name=config.EMBEDDING_MODEL_NAME, device=config.DEVICE)
        self.vector_store = VectorStore(
            dimension=self.embedder.dimension,
            index_file=config.INDEX_FILE, 
            id_map_file=config.ID_MAP_FILE 
        )
        self.cross_encoder = None
        self.products_df_for_rerank = None 
        self.all_products_details = None 

        if self.vector_store.index is None:
            logger.error("Vector store index is not loaded or built. Cannot perform search.")
            raise RuntimeError("Failed to load vector store index.")

        try:
            logger.info(f"Loading ALL product data for locale '{locale}'...")
            self.all_products_details = load_products(locale=locale)
            self.all_products_details['product_text'] = self.all_products_details.apply(prepare_text_for_product, axis=1)
            self.products_df_for_rerank = self.all_products_details[['product_id', 'product_text']].set_index('product_id')
            logger.info(f"Loaded and prepared text for {len(self.products_df_for_rerank)} products.")
        except Exception as e:
            logger.error(f"Failed to load product data: {e}. Re-ranking and detailed output might not be possible.")

        if CrossEncoder:
            load_path = None
            model_description = ""

            if model_type == 'finetuned':
                fine_tuned_path = str(config.FINETUNED_CROSS_ENCODER_PATH)
                if os.path.isdir(fine_tuned_path):
                    logger.info(f"Attempting to load fine-tuned model from: {fine_tuned_path}")
                    load_path = fine_tuned_path
                    model_description = "Fine-tuned Model"
                else:
                    logger.warning(f"Fine-tuned model path not found: {fine_tuned_path}. Falling back to base model.")
                    load_path = config.BASE_CROSS_ENCODER_MODEL
                    model_description = "Base Model (Fine-tuned not found)"
            elif model_type == 'base':
                 load_path = config.BASE_CROSS_ENCODER_MODEL
                 model_description = "Base Model"
            else:
                 logger.warning(f"Invalid model_type '{model_type}'. Defaulting to base model.")
                 load_path = config.BASE_CROSS_ENCODER_MODEL
                 model_description = "Base Model (Invalid type specified)"

            if load_path:
                logger.info(f"Loading cross-encoder: {model_description} ({load_path})")
                try:
                    # Add max_length from previous fix
                    self.cross_encoder = CrossEncoder(
                        load_path,
                        device=config.DEVICE,
                        max_length=512 # Consistent truncation
                    )
                    logger.info(f"{model_description} cross-encoder loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load {model_description} cross-encoder from {load_path}: {e}")
                    self.cross_encoder = None # Ensure it's None if loading fails
        else:
             logger.warning("CrossEncoder class not available. Re-ranking disabled.")

    @timeit
    def perform_search(self, query: str, k_initial: int, k_final: int, rerank: bool):
        """
        Performs semantic search, optionally followed by cross-encoder re-ranking.

        Args:
            query (str): The user's search query.
            k_initial (int): Number of candidates to retrieve from initial semantic search.
            k_final (int): Final number of results to return.
            rerank (bool): Whether to use the cross-encoder for re-ranking.

        Returns:
            tuple: A tuple containing:
                - list[str]: List of final product IDs.
                - list[float]: List of corresponding final scores.
                Returns ([], []) if search fails.
        """
        if not query:
            logger.warning("Search query is empty.")
            return [], []

        if self.vector_store.index is None:
             logger.error("Search failed: Vector index not available.")
             return [], []

        # Determine if re-ranking is possible and requested
        use_reranking = rerank and self.cross_encoder and (self.products_df_for_rerank is not None)
        if rerank and not use_reranking:
             logger.warning("Re-ranking requested but prerequisites (model loaded, product data available) not met. Performing semantic search only.")

        num_candidates_to_fetch = k_initial if use_reranking else k_final

        # --- Stage 1: Initial Semantic Retrieval ---
        logger.info(f"Performing initial semantic search for top {num_candidates_to_fetch} candidates...")
        query_embedding = self.embedder.encode([query], show_progress_bar=False)
        if query_embedding is None or len(query_embedding) == 0:
            logger.error("Failed to generate query embedding.")
            return [], []

        initial_ids, initial_scores = self.vector_store.search(query_embedding[0], k=num_candidates_to_fetch)

        if not initial_ids:
            logger.info("Initial semantic search returned no results.")
            return [], []

        # --- Stage 2: Re-ranking (Optional) ---
        if use_reranking:
            logger.info(f"Performing re-ranking for {len(initial_ids)} candidates...")
            rerank_pairs = []
            valid_ids_for_rerank = []
            for pid in initial_ids:
                 try:
                     # Use the pre-loaded lookup table
                     product_text = self.products_df_for_rerank.loc[pid, 'product_text']
                     rerank_pairs.append([query, product_text])
                     valid_ids_for_rerank.append(pid)
                 except KeyError:
                      logger.warning(f"Product ID {pid} from initial search not found in loaded product data. Skipping for re-ranking.")

            if not rerank_pairs:
                 logger.warning("No valid candidates remaining after checking product data. Returning top results from initial search.")
                 final_ids = initial_ids[:k_final]
                 final_scores = initial_scores[:k_final]
            else:
                try:
                    logger.info(f"Predicting scores for {len(rerank_pairs)} pairs...")
                    cross_scores = self.cross_encoder.predict(rerank_pairs, show_progress_bar=True) # Show progress for reranking
                    logger.info(f"Generated {len(cross_scores)} re-ranking scores.")

                    reranked_results = sorted(zip(valid_ids_for_rerank, cross_scores), key=lambda x: x[1], reverse=True)

                    final_ids = [pid for pid, score in reranked_results[:k_final]]
                    final_scores = [float(score) for pid, score in reranked_results[:k_final]]
                    logger.info(f"Re-ranked results generated.")

                except Exception as e:
                    logger.error(f"Error during cross-encoder prediction or processing: {e}. Falling back to initial semantic search results.")
                    final_ids = initial_ids[:k_final]
                    final_scores = initial_scores[:k_final]
        else:
            # Not re-ranking, just take top k_final from initial results
            logger.info("Skipping re-ranking step.")
            final_ids = initial_ids[:k_final]
            final_scores = initial_scores[:k_final]

        return final_ids, final_scores

    def get_product_details(self, product_ids: list) -> pd.DataFrame:
        """Retrieves full product details for a list of product IDs."""
        if self.all_products_details is None:
            logger.warning("Product details DataFrame not loaded. Cannot retrieve details.")
            return pd.DataFrame() # Return empty DataFrame

        try:
            # Ensure index is set for efficient lookup
            if self.all_products_details.index.name != 'product_id':
                 details_df = self.all_products_details.set_index('product_id')
            else:
                 details_df = self.all_products_details

            # Select rows, preserving order of input IDs
            results_df = details_df.loc[product_ids].reset_index()
             # Reorder results DF to match input product_ids list if necessary
            results_df['product_id'] = pd.Categorical(results_df['product_id'], categories=product_ids, ordered=True)
            results_df = results_df.sort_values('product_id')

            return results_df 

        except KeyError as e:
             logger.error(f"Some product IDs not found in details data: {e}")
             found_ids = [pid for pid in product_ids if pid in details_df.index]
             if found_ids:
                 results_df = details_df.loc[found_ids].reset_index()
                 results_df['product_id'] = pd.Categorical(results_df['product_id'], categories=product_ids, ordered=True)
                 results_df = results_df.sort_values('product_id')
                 return results_df
             else:
                 return pd.DataFrame()
        except Exception as e:
             logger.error(f"Error retrieving product details: {e}")
             return pd.DataFrame()


# --- Command-Line Execution Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform semantic search with optional re-ranking.")
    parser.add_argument("query", type=str, help="The search query.")
    parser.add_argument(
        "--k_initial", type=int, default=config.RERANK_CANDIDATE_COUNT,
        help=f"Number of candidates for initial semantic search (default: {config.RERANK_CANDIDATE_COUNT})."
    )
    parser.add_argument(
        "--k_final", type=int, default=config.DEFAULT_SEARCH_K,
        help=f"Final number of results to return (default: {config.DEFAULT_SEARCH_K})."
    )
    parser.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, default=True, # Use --rerank or --no-rerank
        help="Enable cross-encoder re-ranking (default: enabled)."
    )
    parser.add_argument(
        "--model_type", choices=['finetuned', 'base'], default='finetuned',
        help="Which cross-encoder model to use if re-ranking ('finetuned' or 'base', default: 'finetuned')."
    )
    parser.add_argument(
        "--locale", type=str, default="us",
        help="Locale for product data (default: 'us')."
    )

    args = parser.parse_args()

    logger.info(f"Search Query: '{args.query}'")
    logger.info(f"Initial Candidates (k_initial): {args.k_initial}")
    logger.info(f"Final Results (k_final): {args.k_final}")
    logger.info(f"Re-ranking Enabled: {args.rerank}")
    if args.rerank:
        logger.info(f"Re-ranker Model Type: {args.model_type}")
    logger.info(f"Locale: {args.locale}")
    logger.info("-" * 30)


    try:
        search_engine = SemanticSearchEngine(locale=args.locale, model_type=args.model_type if args.rerank else 'base')
        final_product_ids, final_scores = search_engine.perform_search(
            query=args.query,
            k_initial=args.k_initial,
            k_final=args.k_final,
            rerank=args.rerank
        )

        # Print results
        logger.info("\n--- Search Results ---")
        if final_product_ids:
            # Get details for the final IDs
            details_df = search_engine.get_product_details(final_product_ids)

            # Combine details with scores
            results_map = {pid: score for pid, score in zip(final_product_ids, final_scores)}
            details_df['score'] = details_df['product_id'].map(results_map)

            # Select columns to display and reorder
            display_columns = ['product_id', 'score', 'product_title', 'product_brand', 'product_color']
            display_columns.append('product_description')
            display_columns.append('product_text')

            display_df = details_df[[col for col in display_columns if col in details_df.columns]]

            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                 print(display_df.to_string(index=False))



        else:
            print("No results found.")

    except RuntimeError as e:
         logger.error(f"Search engine initialization failed: {e}")
    except Exception as e:
         logger.error(f"An unexpected error occurred during search: {e}", exc_info=True) # Log full traceback