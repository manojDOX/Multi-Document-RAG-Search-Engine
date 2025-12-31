from typing import Generator, Optional, List
import streamlit as st

from core.ingestion import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.chain import RAGChain
from tools.tavily_search import TavilySearchTool
from ui.components import save_uploaded_file


class ChatInterface:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        self.rag_chain: Optional[RAGChain] = None
        self.tavily = TavilySearchTool()

    # ==========================================================
    # DOCUMENT INGESTION (EXPLICIT BUTTON DRIVEN)
    # ==========================================================
    def process_uploaded_files(self, uploaded_files) -> int:
        all_chunks = []

        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            documents = self.doc_processor.process(file_path)

            for idx, doc in enumerate(documents):
                doc.metadata.update({
                    "source": uploaded_file.name,
                    "chunk_id": idx,
                    "source_type": "doc"
                })

            all_chunks.extend(documents)

            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)

        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            st.session_state.vector_store_initialized = True

        return len(all_chunks)

    # ==========================================================
    # RAG INITIALIZATION
    # ==========================================================
    def initialize_rag_chain(self):
        if self.vector_store.is_initialized and self.rag_chain is None:
            self.rag_chain = RAGChain(self.vector_store)

    # ==========================================================
    # RESPONSE GENERATION (USER-SELECTED MODE)
    # ==========================================================
    def get_response(
        self,
        query: str,
        retrieval_mode: str
    ) -> Generator[str, None, None]:

        if self.rag_chain is None and self.vector_store.is_initialized:
            self.initialize_rag_chain()

        st.session_state.last_answer_meta = {
            "answer_type": retrieval_mode,
            "doc_chunks": [],
            "web_docs": [],
            "doc_summaries": []
        }

        # ---------------- DOCUMENT ----------------
        if retrieval_mode == "doc":
            if not self.vector_store.is_initialized:
                yield "❗ Please process documents first."
                return

            docs = self.vector_store.search(query)
            st.session_state.last_answer_meta["doc_chunks"] = docs

            summaries = self.rag_chain.summarize_documents(docs, top_n=3)
            st.session_state.last_answer_meta["doc_summaries"] = summaries

            for token in self.rag_chain.query_stream(query):
                yield token
            return

        # ---------------- WEB ----------------
        if retrieval_mode == "web":
            web_docs = self.tavily.as_documents(query)
            st.session_state.last_answer_meta["web_docs"] = web_docs

            context = "\n\n".join(
                f"[Web] {d.metadata.get('title', 'Unknown')}\n{d.page_content}"
                for d in web_docs
            )

            answer = self.rag_chain.generate(query, context)
            yield answer
            return

        # ---------------- HYBRID ----------------
        if retrieval_mode == "hybrid":
            docs = []
            if self.vector_store.is_initialized:
                docs = self.vector_store.search(query)

            web_docs = self.tavily.as_documents(query)

            st.session_state.last_answer_meta["doc_chunks"] = docs
            st.session_state.last_answer_meta["web_docs"] = web_docs

            context_parts = []

            for d in docs[:3]:
                context_parts.append(
                    f"[Doc] {d.metadata.get('source')} – Chunk{d.metadata.get('chunk_id')}\n{d.page_content}"
                )

            for w in web_docs[:3]:
                context_parts.append(
                    f"[Web] Tavily: {w.metadata.get('title', 'Unknown')}\n{w.page_content}"
                )

            context = "\n\n".join(context_parts)

            answer = self.rag_chain.generate(query, context)
            yield answer

    # ==========================================================
    # SOURCE LIST FOR CHAT HISTORY
    # ==========================================================
    def get_sources(self, query: str, retrieval_mode: str) -> List[str]:
        sources = []

        if retrieval_mode in ("doc", "hybrid") and self.vector_store.is_initialized:
            docs = self.vector_store.search(query)
            sources.extend(
                {f"[Doc] {d.metadata.get('source')}" for d in docs}
            )

        if retrieval_mode in ("web", "hybrid"):
            sources.append("[Web] Tavily Search")

        return list(sources)
