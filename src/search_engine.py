import logging
import config
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from utils import timeit
from data_loader import load_products
import numpy as np
import os
import pandas as pd
import sys  # Added for command-line argument handling

# Optional LangChain components (uncomment if using)
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI # Or other LLM provider

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """Orchestrates embedding generation and vector search."""
    def __init__(self):
        logger.info("Initializing Semantic Search Engine...")
        self.embedder = EmbeddingGenerator()  # Uses model from config
        # Initialize VectorStore - dimension is derived from the loaded embedder
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.products_df = load_products()
        if self.vector_store.index is None:
            logger.warning(
                "Vector store index is not loaded or built. "
                "Search functionality will be unavailable until 'generate_embeddings.py' is run."
            )

    # --- Optional: LangChain Query Refinement Setup ---
    # Example: Setup a simple chain to potentially rephrase the query
    # self.llm = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY) # Requires API key
    # self.refine_prompt = PromptTemplate.from_template(
    #     "Rephrase the following user query for better e-commerce product search:\n"
    #     "Query: {query}\n"
    #     "Rephrased Query:"
    # )
    # self.refine_chain = {"query": RunnablePassthrough()} | self.refine_prompt | self.llm
    # logger.info("LangChain query refinement chain initialized (optional).")
    # ----------------------------------------------------

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
                - list[dict]: List of retrieved product details (each as a dictionary).
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
        if refine_query:
            # --- Optional: LangChain Query Refinement ---
            # try:
            #     logger.info(f"Refining query: '{query}'")
            #     # Note: LangChain outputs might need parsing (e.g., AIMessage content)
            #     response = self.refine_chain.invoke(query)
            #     # Assuming response is a string or has a 'content' attribute
            #     processed_query = getattr(response, 'content', str(response)).strip()
            #     logger.info(f"Refined query: '{processed_query}'")
            # except Exception as e:
            #     logger.error(f"LangChain query refinement failed: {e}. Using original query.")
            #     processed_query = query
            # ---------------------------------------------
            logger.warning("Query refinement requested but not fully implemented in this example.")
            processed_query = query  # Fallback for now

        logger.info(f"Encoding query: '{processed_query}'")
        query_embedding = self.embedder.encode([processed_query], show_progress_bar=False)

        if query_embedding is None or len(query_embedding) == 0:
            logger.error("Failed to generate query embedding.")
            return [], []

        # Perform search in the vector store
        results, scores = self.vector_store.search(query_embedding[0], k=k)

        # Get full product details for the results
        product_details = []
        for pid in results:
            product_info = self.products_df[self.products_df['product_id'] == pid].iloc[0].to_dict()
            product_details.append(product_info)

        return product_details, scores


# Example usage (for testing)
if __name__ == '__main__':
    logger.info("Testing SemanticSearchEngine...")

    # Fetch test_query from command-line args (default if none passed)
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "Mini coffee machine with pods black color"

    # This test assumes an index has been built (e.g., by running generate_embeddings.py or the test in vector_store.py)
    # For a standalone test, you might need to manually ensure the test index exists.
    # Let's try using the test index created by vector_store.py if run directly
    try:
        DIM = 5  # Must match the dimension used in vector_store.py test
        test_embeddings = np.random.rand(10, DIM).astype(np.float32)
        test_ids = [f"prod_{i}" for i in range(10)]
        test_index_file = os.path.join(config.INDEX_DIR, "test_index.faiss")
        test_id_map_file = os.path.join(config.INDEX_DIR, "test_id_map.json")

        # Ensure dummy index exists for test
        if not os.path.exists(test_index_file) or not os.path.exists(test_id_map_file):
            print("Creating dummy index for search engine test...")
            vs = VectorStore(dimension=DIM, index_file=test_index_file, id_map_file=test_id_map_file)
            vs.build_index(test_embeddings, test_ids)
            del vs  # Release file handles

        # Temporarily override config for the test
        original_index_file = config.INDEX_FILE
        original_id_map_file = config.ID_MAP_FILE
        config.INDEX_FILE = test_index_file
        config.ID_MAP_FILE = test_id_map_file
        config.EMBEDDING_MODEL_NAME = 'paraphrase-MiniLM-L3-v2'  # A small model for quick test

        print("Initializing engine (might download model)...")
        search_engine = SemanticSearchEngine()

        if search_engine.vector_store.index:
            print(f"\nPerforming search for: '{test_query}'")
            results, scores = search_engine.perform_search(test_query, k=5)
            print(f"Results: {results}")
            results_df = pd.DataFrame(results)
            print(results_df)
            print(f"Scores: {scores}")
        else:
            print("\nCould not perform search test, index not loaded.")

        # Clean up test files
        if os.path.exists(test_index_file):
            os.remove(test_index_file)
        if os.path.exists(test_id_map_file):
            os.remove(test_id_map_file)

        # Restore original config paths
        config.INDEX_FILE = original_index_file
        config.ID_MAP_FILE = original_id_map_file

    except ImportError as ie:
        print(f"Import error during test: {ie}. Make sure all dependencies are installed.")
    except Exception as e:
        print(f"An error occurred during the search engine test: {e}")

