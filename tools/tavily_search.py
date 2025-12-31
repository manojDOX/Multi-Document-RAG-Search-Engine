import os
from typing import List
from langchain_tavily import TavilySearch
from langchain_core.documents import Document

from config.settings import settings


class TavilySearchTool:
    """
    This class provides web search capability using the Tavily API.
    It converts search results into Document objects that can be
    used directly in retrieval workflows.
    """

    def __init__(self, max_results: int = 3):
        """
        Initializes the Tavily search tool.

        Args:
            max_results: Maximum number of search results to return.

        Returns:
            None
        """
        os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
        self.search = TavilySearch(max_results=max_results)

    def as_documents(self, query: str) -> List[Document]:
        """
        Executes a web search and converts results into documents.

        Args:
            query: Search query string.

        Returns:
            A list of Document objects created from web search results.
        """
        results = self.search.invoke(query)
        docs = []

        for r in results.get("results", []):
            docs.append(
                Document(
                    page_content=r.get("content", ""),
                    metadata={
                        "source_type": "web",
                        "title": r.get("title", "Unknown"),
                        "source": r.get("url")
                    }
                )
            )
        return docs
