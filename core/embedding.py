from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


class EmbeddingManager:
    """
    Manages text embeddings using HuggingFace models.
    
    Uses sentence-transformers which are FREE and run locally!
    No API costs for embeddings.
    """
    _embedding_model: HuggingFaceEmbeddings = None
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: HuggingFace model name (default from settings)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        
        # Initialize HuggingFace embeddings (downloads model on first use)
        if EmbeddingManager._embedding_model is None:
            EmbeddingManager._embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
    
    @property
    def model(self) -> HuggingFaceEmbeddings:
        """Get the embeddings model instance."""
        return self._embedding_model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        
        return self.model.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.model.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Integer dimension size
        """
        # Create a sample embedding to get dimension
        return len(self.embed_text("dimension_probe"))