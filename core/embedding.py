from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


class EmbeddingManager:
    """
    This class handles text embedding generation.
    It uses a HuggingFace embedding model to convert text
    into numerical vectors that can be used for search
    and similarity operations.
    """

    _embedding_model: HuggingFaceEmbeddings = None

    def __init__(self, model_name: str = None):
        """
        Initializes the embedding manager with a HuggingFace model.

        Args:
            model_name: Optional name of the embedding model.
                        If not provided, the value from settings is used.

        Returns:
            None
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL

        if EmbeddingManager._embedding_model is None:
            EmbeddingManager._embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """
        Provides access to the embedding model instance.

        Args:
            None

        Returns:
            The initialized HuggingFaceEmbeddings model.
        """
        return self._embedding_model

    def embed_text(self, text: str) -> List[float]:
        """
        Converts a single text string into an embedding vector.

        Args:
            text: The input text to be converted into an embedding.

        Returns:
            A list of float values representing the embedding.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        return self.model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Converts multiple text strings into embedding vectors.

        Args:
            texts: A list of input texts.

        Returns:
            A list of embedding vectors, one for each text.
        """
        return self.model.embed_documents(texts)

    def get_embedding_dimension(self) -> int:
        """
        Returns the size of the embedding vector.

        Args:
            None

        Returns:
            An integer representing the embedding dimension.
        """
        return len(self.embed_text("dimension_probe"))
