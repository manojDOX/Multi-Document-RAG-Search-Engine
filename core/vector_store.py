"""
Vector Store Module
===================
DAY 2: This module handles FAISS vector database operations.

SOLID Principle: Single Responsibility Principle (SRP)
- This class has ONE job: manage the vector store

Topics to teach:
- What is a vector database?
- FAISS (Facebook AI Similarity Search)
- Indexing documents
- Similarity search
- Persistence (save/load)
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from core.embedding import EmbeddingManager


class VectorStoreManager:
    """
    Manages FAISS vector store operations.

    FAISS is a FREE, local vector database that runs entirely on your machine.
    No external services or API costs!
    """

    def __init__(self, embedding_manager: EmbeddingManager = None):
        """
        Initialize the vector store manager.

        Args:
            embedding_manager: EmbeddingManager instance (creates one if not provided)
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._vector_store: Optional[FAISS] = None
        self.index_path = settings._FAISS_INDEX_PATH

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------
    @property
    def vector_store(self) -> Optional[FAISS]:
        """Return the FAISS vector store instance."""
        return self._vector_store

    @property
    def is_initialized(self) -> bool:
        """Check whether the vector store is initialized."""
        return self._vector_store is not None

    # --------------------------------------------------
    # Index creation & updates
    # --------------------------------------------------
    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Create a new FAISS vector store from documents.

        Args:
            documents: List of Document objects to index

        Returns:
            FAISS vector store instance
        """
        documents = [doc for doc in documents if doc.page_content.strip()]

        self._vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_manager.model
        )
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to an existing vector store.
        Creates a new one if not initialized.
        """
        documents = [doc for doc in documents if doc.page_content.strip()]

        if not self.is_initialized:
            self.create_from_documents(documents)
        else:
            self._vector_store.add_documents(documents)

    # --------------------------------------------------
    # Search
    # --------------------------------------------------
    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Search query
            k: Number of results (default from settings)

        Returns:
            List of relevant Document objects
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Add documents first.")

        k = k or settings.TOP_K_RESULTS
        return self._vector_store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = None) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Add documents first.")

        k = k or settings.TOP_K_RESULTS
        return self._vector_store.similarity_search_with_score(query, k=k)

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
    def save(self, path: str = None) -> None:
        """
        Save FAISS index to disk.
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Nothing to save.")

        save_path = path or self.index_path
        os.makedirs(save_path, exist_ok=True)
        self._vector_store.save_local(save_path)

    def load(self, path: str = None) -> FAISS:
        """
        Load FAISS index from disk.

        NOTE:
        allow_dangerous_deserialization=True is required for FAISS
        because it uses pickle internally. This is SAFE for trusted
        local files created by this application.
        """
        load_path = path or self.index_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No saved FAISS index found at {load_path}")

        self._vector_store = FAISS.load_local(
            load_path,
            self.embedding_manager.model,
            allow_dangerous_deserialization=True
        )
        return self._vector_store

    # --------------------------------------------------
    # Retriever interface
    # --------------------------------------------------
    def get_retriever(self, k: int = None):
        """
        Return a LangChain-compatible retriever.
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized.")

        k = k or settings.TOP_K_RESULTS
        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    def clear(self) -> None:
        """Clear vector store from memory."""
        self._vector_store = None
