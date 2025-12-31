import os
from typing import List
from langchain_tavily import TavilySearch
from langchain_core.documents import Document

from config.settings import settings


class TavilySearchTool:
    def __init__(self, max_results: int = 3):
        os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
        self.search = TavilySearch(max_results=max_results)

    def as_documents(self, query: str) -> List[Document]:
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
