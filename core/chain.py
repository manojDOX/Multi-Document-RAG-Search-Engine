from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from config.settings import settings


RAG_PROMPT = """You are a helpful assistant.
Answer the question ONLY using the context below.
If the answer is not present, say so clearly.

Context:
{context}

Question: {question}

Answer:
"""


class RAGChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store

        self.llm = ChatGroq(
            model=settings.GPT_MODEL_NAME,
            temperature=settings.TEMPRATURE,
            api_key=settings.GROQ_API_KEY
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        self.parser = StrOutputParser()

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        return self.vector_store.search(query, k=k)

    def generate(self, query: str, context: str) -> str:
        chain = self.prompt | self.llm | self.parser
        return chain.invoke({"context": context, "question": query})

    def query_stream(self, query: str):
        docs = self.retrieve(query)
        context = "\n\n".join(
            f"[Doc] {d.metadata.get('source')} – Chunk{i+1}\n{d.page_content}"
            for i, d in enumerate(docs)
        )

        chain = self.prompt | self.llm | self.parser
        for token in chain.stream({"context": context, "question": query}):
            yield token

    def summarize_documents(self, documents: List[Document], top_n: int = 3):
        summaries = []
        for i, doc in enumerate(documents[:top_n], 1):
            prompt = f"Summarize briefly:\n{doc.page_content}"
            summary = self.llm.invoke(prompt).content
            summaries.append(
                f"[Doc] {doc.metadata.get('source')} – Chunk{i}\n{summary}"
            )
        return summaries
