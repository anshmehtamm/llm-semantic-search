import faiss
import numpy as np
import logging
import os
import config
from utils import timeit, save_json, load_json

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages the FAISS vector index and ID mapping."""
    def __init__(self, dimension, index_file=config.INDEX_FILE, id_map_file=config.ID_MAP_FILE):
        self.dimension = dimension
        self.index_file = index_file
        self.id_map_file = id_map_file
        self.index = None
        self.id_map = {} 

        self._load_index()

    @timeit
    def build_index(self, embeddings: np.ndarray, product_ids: list):
        """
        Builds the FAISS index from embeddings and saves it along with the ID map.

        Args:
            embeddings (np.ndarray): A 2D numpy array of embeddings.
            product_ids (list): A list of product IDs corresponding to the embeddings.
        """
        if embeddings is None or len(embeddings) == 0:
            logger.error("Cannot build index with empty embeddings.")
            return
        if len(embeddings) != len(product_ids):
            logger.error(f"Mismatch between number of embeddings ({len(embeddings)}) and product IDs ({len(product_ids)}).")
            return

        logger.info(f"Building FAISS index with {len(embeddings)} vectors of dimension {self.dimension}.")

        # Using IndexFlatL2 for simplicity first
        self.index = faiss.IndexFlatL2(self.dimension)

        try:
            self.index.add(embeddings.astype(np.float32)) # FAISS requires float32
            logger.info(f"Added {self.index.ntotal} vectors to the index.")

            # Create the mapping from index position to product ID
            self.id_map = {i: pid for i, pid in enumerate(product_ids)}

            # Save index and map
            self._save_index()

        except Exception as e:
            logger.error(f"Error building or saving FAISS index: {e}")
            self.index = None
            self.id_map = {}

    @timeit
    def search(self, query_embedding: np.ndarray, k: int = config.DEFAULT_SEARCH_K):
        """
        Searches the index for the k nearest neighbors of the query embedding.

        Args:
            query_embedding (np.ndarray): The embedding of the query (1D array).
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            tuple: A tuple containing:
                - list[str]: List of retrieved product IDs.
                - list[float]: List of corresponding distances/scores.
                Returns ([], []) if search fails or index is not ready.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called but index is not built or is empty.")
            return [], []
        if query_embedding is None or query_embedding.ndim != 1:
             logger.error("Invalid query embedding provided for search.")
             return [], []

        # FAISS expects a 2D array for queries, even if it's just one
        query_embedding_2d = np.expand_dims(query_embedding.astype(np.float32), axis=0)

        try:
            # Perform the search
            distances, indices = self.index.search(query_embedding_2d, k)

            # Process results
            retrieved_indices = indices[0] # Get results for the first (only) query
            retrieved_distances = distances[0]

            # Map indices back to product IDs
            retrieved_ids = [self.id_map.get(idx) for idx in retrieved_indices if idx != -1 and idx in self.id_map]
            # Filter distances corresponding to valid IDs
            valid_distances = [dist for idx, dist in zip(retrieved_indices, retrieved_distances) if idx != -1 and idx in self.id_map]


            if not retrieved_ids:
                logger.info("Search returned no valid results.")
                return [], []

            return retrieved_ids, valid_distances

        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            return [], []

    @timeit
    def _save_index(self):
        """Saves the FAISS index and the ID map to disk."""
        if self.index is not None:
            try:
                faiss.write_index(self.index, self.index_file)
                logger.info(f"FAISS index saved to {self.index_file}")
            except Exception as e:
                logger.error(f"Error saving FAISS index to {self.index_file}: {e}")
        else:
            logger.warning("Attempted to save index, but it's not built.")

        if self.id_map:
            save_json(self.id_map, self.id_map_file)
        else:
            logger.warning("Attempted to save ID map, but it's empty.")


    @timeit
    def _load_index(self):
        """Loads the FAISS index and ID map from disk if they exist."""
        index_exists = os.path.exists(self.index_file)
        map_exists = os.path.exists(self.id_map_file)

        if index_exists and map_exists:
            logger.info(f"Loading existing index from {self.index_file} and map from {self.id_map_file}")
            try:
                self.index = faiss.read_index(self.index_file)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")

                # Load the ID map, converting keys back to integers
                loaded_map_str_keys = load_json(self.id_map_file)
                if loaded_map_str_keys:
                     self.id_map = {int(k): v for k, v in loaded_map_str_keys.items()}
                     logger.info(f"Loaded ID map with {len(self.id_map)} entries.")
                     # Basic validation
                     if self.index.ntotal != len(self.id_map):
                         logger.warning(f"Index size ({self.index.ntotal}) != ID map size ({len(self.id_map)}). Possible inconsistency.")
                else:
                    logger.error("Failed to load ID map, although file exists.")
                    self.index = None 

            except Exception as e:
                logger.error(f"Error loading index or map: {e}. Index will need to be rebuilt.")
                self.index = None
                self.id_map = {}
        elif index_exists != map_exists:
             logger.warning(f"Index file exists ({index_exists}) but ID map file exists ({map_exists}). Files seem inconsistent. Index will need to be rebuilt.")
             self.index = None
             self.id_map = {}
        else:
            logger.info("No existing index found. Index needs to be built.")
            self.index = None
            self.id_map = {}

# Example usage (for testing, assumes embeddings generated elsewhere)
if __name__ == '__main__':
    logger.info("Testing VectorStore...")
    DIM = 5 # Small dimension for testing
    test_embeddings = np.random.rand(10, DIM).astype(np.float32)
    test_ids = [f"prod_{i}" for i in range(10)]
    test_index_file = os.path.join(config.INDEX_DIR, "test_index.faiss")
    test_id_map_file = os.path.join(config.INDEX_DIR, "test_id_map.json")

    # --- Test Building ---
    vector_store = VectorStore(dimension=DIM, index_file=test_index_file, id_map_file=test_id_map_file)
    vector_store.build_index(test_embeddings, test_ids)

    # --- Test Searching ---
    if vector_store.index:
        query_vec = np.random.rand(DIM).astype(np.float32)
        results, distances = vector_store.search(query_vec, k=3)
        print(f"\nSearch Results (IDs): {results}")
        print(f"Search Distances: {distances}")

    # --- Test Loading ---
    print("\nTesting loading from file...")
    vector_store_loaded = VectorStore(dimension=DIM, index_file=test_index_file, id_map_file=test_id_map_file)
    if vector_store_loaded.index:
        query_vec_2 = test_embeddings[0] # Search for something known
        results_loaded, distances_loaded = vector_store_loaded.search(query_vec_2, k=3)
        print(f"\nLoaded Index Search Results (IDs): {results_loaded}")
        print(f"Loaded Index Search Distances: {distances_loaded}")
        # The first result should ideally be 'prod_0' with distance ~0
        assert results_loaded[0] == 'prod_0'
        assert np.isclose(distances_loaded[0], 0.0)
        print("Loading test passed.")
    else:
        print("Loading test failed: Index not loaded.")

    # Clean up test files
    if os.path.exists(test_index_file): os.remove(test_index_file)
    if os.path.exists(test_id_map_file): os.remove(test_id_map_file) 