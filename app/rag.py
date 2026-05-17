from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    openrouter_api_key: str
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    pinecone_index: str = "cv-rag"
    openrouter_model: str = "mistralai/mistral-7b-instruct:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    retriever_top_k: int = 5


def get_settings() -> Settings:
    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not pinecone_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment")
    if not openrouter_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment")

    return Settings(
        pinecone_api_key=pinecone_key,
        openrouter_api_key=openrouter_key,
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        pinecone_index=os.getenv("PINECONE_INDEX", "cv-rag"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        retriever_top_k=int(os.getenv("RETRIEVER_TOP_K", "5")),
    )


class PineconeRetriever:
    """Small Pinecone retriever compatible with LangChain-style RAG chains."""

    def __init__(self, index: Any, embedding_model: SentenceTransformer, top_k: int = 5):
        self.index = index
        self.embedding_model = embedding_model
        self.top_k = top_k

    def invoke(self, query: str) -> List[Document]:
        embedding = self.embedding_model.encode(query).tolist()
        response = self.index.query(
            vector=embedding,
            top_k=self.top_k,
            include_metadata=True,
        )

        documents: List[Document] = []
        for match in response.get("matches", []):
            metadata = match.get("metadata") or {}
            text = (
                metadata.get("text")
                or metadata.get("chunk_text")
                or metadata.get("page_content")
                or metadata.get("content")
                or ""
            )
            if not text:
                continue

            doc_metadata = dict(metadata)
            doc_metadata["score"] = match.get("score")
            doc_metadata["id"] = match.get("id")
            documents.append(Document(page_content=text, metadata=doc_metadata))

        return documents


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant CV context was retrieved."

    formatted_chunks: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source") or doc.metadata.get("file_name") or doc.metadata.get("name") or "unknown"
        score = doc.metadata.get("score")
        score_text = f" | score={score:.4f}" if isinstance(score, float) else ""
        formatted_chunks.append(
            f"[CV Chunk {idx} | source={source}{score_text}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)


PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful assistant that answers questions using ONLY the CV context below.

Rules:
- If the answer is not available in the context, say that you do not know based on the CVs.
- Be concise, accurate, and specific.
- Mention names, skills, experience, education, or projects only when they appear in the context.
- Do not invent facts.

CV context:
{context}

Question:
{question}

Answer:
""".strip()
)


settings: Optional[Settings] = None
retriever: Optional[PineconeRetriever] = None
rag_chain: Optional[Any] = None
llm: Optional[ChatOpenAI] = None


def init_rag() -> None:
    """Initialize Pinecone, embedding model, LLM, and RAG chain once at startup."""
    global settings, retriever, rag_chain, llm

    settings = get_settings()

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index)
    embedding_model = SentenceTransformer(settings.embedding_model)

    retriever = PineconeRetriever(
        index=index,
        embedding_model=embedding_model,
        top_k=settings.retriever_top_k,
    )

    llm = ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
    )

    rag_chain = (
        {
            "context": RunnableLambda(lambda question: format_docs(retriever.invoke(question))),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )


def ensure_initialized() -> None:
    if rag_chain is None or retriever is None:
        init_rag()


def run_query(question: str, include_context: bool = False) -> Dict[str, Any]:
    ensure_initialized()
    assert rag_chain is not None
    assert retriever is not None

    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = rag_chain.invoke(question)

    result: Dict[str, Any] = {
        "question": question,
        "answer": answer,
    }

    if include_context:
        result["context"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        result["formatted_context"] = context

    return result


async def run_query_stream(question: str) -> AsyncGenerator[str, None]:
    ensure_initialized()
    assert rag_chain is not None

    async for chunk in rag_chain.astream(question):
        if chunk:
            yield chunk
