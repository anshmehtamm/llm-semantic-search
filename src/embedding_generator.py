import torch
from sentence_transformers import SentenceTransformer
import logging
import config
from utils import timeit

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles loading the embedding model and generating embeddings."""
    def __init__(self, model_name=config.EMBEDDING_MODEL_NAME, device=config.DEVICE):
        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")
        self.model = self._load_model(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model '{model_name}' loaded. Dimension: {self.dimension}")

    def _get_device(self, requested_device):
        """Sets the computation device (CPU or CUDA)."""
        if requested_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif requested_device == "cuda":
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return "cpu"

    @timeit
    def _load_model(self, model_name):
        """Loads the SentenceTransformer model."""
        try:
            model = SentenceTransformer(model_name, device=self.device)
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
            raise

    @timeit
    def encode(self, texts, batch_size=config.EMBEDDING_BATCH_SIZE, show_progress_bar=True):
        """
        Generates embeddings for a list of texts.

        Args:
            texts (list[str]): A list of strings to embed.
            batch_size (int): The batch size for encoding.
            show_progress_bar (bool): Whether to display a progress bar.

        Returns:
            numpy.ndarray: An array of embeddings.
        """
        if not texts:
            logger.warning("Encode called with empty list of texts.")
            return []
        if not self.model:
             logger.error("Embedding model not loaded.")
             return []

        logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}...")
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                device=self.device, 
                convert_to_numpy=True 
            )
            logger.info(f"Successfully generated {len(embeddings)} embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            return []

if __name__ == '__main__':
    logger.info("Testing EmbeddingGenerator...")
    generator = EmbeddingGenerator()
    test_texts = ["This is the first sentence.", "Here is another example document."]
    embeddings = generator.encode(test_texts)
    if embeddings is not None and len(embeddings) > 0:
        print(f"\nGenerated embeddings shape: {embeddings.shape}")
        print(f"First embedding vector (first 5 dims): {embeddings[0][:5]}")
    else:
        print("\nEmbedding generation failed.") 