from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


class DocumentProcessor:
    """
    This class is responsible for loading documents and splitting their
    content into smaller chunks so they can be used efficiently in
    retrieval and search workflows.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initializes the document processor with chunk configuration.

        Args:
            chunk_size: Maximum size of each text chunk.
            chunk_overlap: Number of overlapping characters between chunks.

        Returns:
            None
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        Loads a document from the given file path.

        Args:
            file_path: Path to the document file.

        Returns:
            A list of loaded Document objects.

        Raises:
            ValueError: If the file type is not supported.
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif extension == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}. Use .txt or .pdf")

        return loader.load()

    def load_from_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Creates a document object from raw text input.

        Args:
            text: Raw text content.
            metadata: Optional metadata dictionary.

        Returns:
            A list containing a single Document object.
        """
        metadata = metadata or {}
        return [Document(page_content=text, metadata=metadata)]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into smaller chunks and assigns chunk IDs.

        Args:
            documents: List of documents to split.

        Returns:
            A list of chunked Document objects.
        """
        chunks = self.text_splitter.split_documents(documents)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx

        return chunks

    def process(self, file_path: str) -> List[Document]:
        """
        Loads a document from a file and splits it into chunks.

        Args:
            file_path: Path to the document file.

        Returns:
            A list of processed and chunked Document objects.
        """
        documents = self.load_document(file_path)

        for doc in documents:
            doc.metadata.update({
                "source": Path(file_path).name,
                "source_type": Path(file_path).suffix.replace(".", "")
            })

        return self.split_documents(documents)

    def process_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Processes raw text by creating and splitting documents.

        Args:
            text: Raw text content.
            metadata: Optional metadata information.

        Returns:
            A list of chunked Document objects.
        """
        documents = self.load_from_text(text, metadata)
        return self.split_documents(documents)
