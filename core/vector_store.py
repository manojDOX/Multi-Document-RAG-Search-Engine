import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from core.embedding import EmbeddingManager


class VectorStoreManager:
    """
    This class manages the FAISS vector store.
    It is responsible for creating, updating, searching,
    saving, and loading vector indexes.
    """

    def __init__(self, embedding_manager: EmbeddingManager = None):
        """
        Initializes the vector store manager.

        Args:
            embedding_manager: Optional embedding manager instance.

        Returns:
            None
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._vector_store: Optional[FAISS] = None
        self.index_path = settings._FAISS_INDEX_PATH

    @property
    def vector_store(self) -> Optional[FAISS]:
        """
        Returns the current vector store instance.

        Args:
            None

        Returns:
            The FAISS vector store or None if not initialized.
        """
        return self._vector_store

    @property
    def is_initialized(self) -> bool:
        """
        Checks whether the vector store is initialized.

        Args:
            None

        Returns:
            True if initialized, otherwise False.
        """
        return self._vector_store is not None

    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Creates a new vector store from documents.

        Args:
            documents: List of documents to index.

        Returns:
            The created FAISS vector store.
        """
        documents = [doc for doc in documents if doc.page_content.strip()]

        self._vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_manager.model
        )
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds documents to the vector store.

        Args:
            documents: List of documents to add.

        Returns:
            None
        """
        documents = [doc for doc in documents if doc.page_content.strip()]

        if not self.is_initialized:
            self.create_from_documents(documents)
        else:
            self._vector_store.add_documents(documents)

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Performs a similarity search.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            A list of matching documents.
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Add documents first.")

        k = k or settings.TOP_K_RESULTS
        return self._vector_store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = None) -> List[tuple]:
        """
        Performs a similarity search with relevance scores.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            A list of document-score tuples.
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Add documents first.")

        k = k or settings.TOP_K_RESULTS
        return self._vector_store.similarity_search_with_score(query, k=k)

    def save(self, path: str = None) -> None:
        """
        Saves the vector store to disk.

        Args:
            path: Optional path to save the index.

        Returns:
            None
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized. Nothing to save.")

        save_path = path or self.index_path
        os.makedirs(save_path, exist_ok=True)
        self._vector_store.save_local(save_path)

    def load(self, path: str = None) -> FAISS:
        """
        Loads the vector store from disk.

        Args:
            path: Optional path to load the index from.

        Returns:
            The loaded FAISS vector store.
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

    def get_retriever(self, k: int = None):
        """
        Returns a retriever interface for the vector store.

        Args:
            k: Number of results to retrieve.

        Returns:
            A LangChain-compatible retriever.
        """
        if not self.is_initialized:
            raise ValueError("Vector store is not initialized.")

        k = k or settings.TOP_K_RESULTS
        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def clear(self) -> None:
        """
        Clears the vector store from memory.

        Args:
            None

        Returns:
            None
        """
        self._vector_store = None
