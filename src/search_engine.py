import logging
from . import config
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .utils import timeit
logger = logging.getLogger(__name__)
import numpy as np
import os

class SemanticSearchEngine:
    """Orchestrates embedding generation and vector search."""
    def __init__(self):
        logger.info("Initializing Semantic Search Engine...")
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore(dimension=self.embedder.dimension)

        if self.vector_store.index is None:
             logger.warning("Vector store index is not loaded or built. Search functionality will be unavailable until 'generate_embeddings.py' is run.")

     

    @timeit
    def perform_search(self, query: str, k: int = config.DEFAULT_SEARCH_K, refine_query: bool = False):
        """
        Performs semantic search for a given query.

        Args:
            query (str): The user's search query.
            k (int): Number of results to return.
            refine_query (bool): Whether to use LangChain to refine the query before embedding.

        Returns:
            tuple: A tuple containing:
                - list[str]: List of retrieved product IDs.
                - list[float]: List of corresponding distances/scores.
                Returns ([], []) if search fails.
        """
        if not query:
            logger.warning("Search query is empty.")
            return [], []

        if self.vector_store.index is None:
             logger.error("Search failed: Vector index not available.")
             return [], []

        processed_query = query


        logger.info(f"Encoding query: '{processed_query}'")
        query_embedding = self.embedder.encode([processed_query], show_progress_bar=False)

        if query_embedding is None or len(query_embedding) == 0:
            logger.error("Failed to generate query embedding.")
            return [], []

        results, scores = self.vector_store.search(query_embedding[0], k=k)

        return results, scores

# Example usage (for testing)
if __name__ == '__main__':
    logger.info("Testing SemanticSearchEngine...")
    try:
        DIM = 5
        test_embeddings = np.random.rand(10, DIM).astype(np.float32)
        test_ids = [f"prod_{i}" for i in range(10)]
        test_index_file = os.path.join(config.INDEX_DIR, "test_index.faiss")
        test_id_map_file = os.path.join(config.INDEX_DIR, "test_id_map.json")

        # Ensure dummy index exists for test
        if not os.path.exists(test_index_file) or not os.path.exists(test_id_map_file):
             print("Creating dummy index for search engine test...")
             vs = VectorStore(dimension=DIM, index_file=test_index_file, id_map_file=test_id_map_file)
             vs.build_index(test_embeddings, test_ids)
             del vs 

        # Temporarily override config for the test
        original_index_file = config.INDEX_FILE
        original_id_map_file = config.ID_MAP_FILE
        config.INDEX_FILE = test_index_file
        config.ID_MAP_FILE = test_id_map_file
        config.EMBEDDING_MODEL_NAME = 'paraphrase-MiniLM-L3-v2' # A small model for quick test

        print("Initializing engine (might download model)...")
        search_engine = SemanticSearchEngine()

        if search_engine.vector_store.index:
            test_query = "example query text"
            print(f"\nPerforming search for: '{test_query}'")
            results, scores = search_engine.perform_search(test_query, k=5)
            print(f"Results: {results}")
            print(f"Scores: {scores}")
        else:
            print("\nCould not perform search test, index not loaded.")

        if os.path.exists(test_index_file): os.remove(test_index_file)
        if os.path.exists(test_id_map_file): os.remove(test_id_map_file)

        config.INDEX_FILE = original_index_file
        config.ID_MAP_FILE = original_id_map_file

    except ImportError as ie:
        print(f"Import error during test: {ie}. Make sure all dependencies are installed.")
    except Exception as e:
        print(f"An error occurred during the search engine test: {e}") 