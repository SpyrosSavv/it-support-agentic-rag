import hashlib
import time
from functools import lru_cache
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings


INDEX_READY_TIMEOUT_SECONDS = 60

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or get_settings().embedding_model).strip().lower()
    try:
        return EMBEDDING_DIMENSIONS[name]
    except KeyError:
        raise ValueError(
            f"Unsupported embedding model '{name}'. Add its dimension to EMBEDDING_DIMENSIONS."
        ) from None


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def ensure_index():
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    name = settings.pinecone_index_name
    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    if pc.has_index(name):
        current_dimension = pc.describe_index(name).dimension
        if current_dimension != desired_dimension:
            raise RuntimeError(
                f"Index '{name}' has dimension {current_dimension}, but '{settings.embedding_model}' "
                f"needs {desired_dimension}. Use a different PINECONE_INDEX_NAME or delete the index manually."
            )
    else:
        pc.create_index(
            name=name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        deadline = time.monotonic() + INDEX_READY_TIMEOUT_SECONDS
        while not pc.describe_index(name).status["ready"]:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Index '{name}' was not ready after {INDEX_READY_TIMEOUT_SECONDS}s"
                )
            time.sleep(1)

    return pc.Index(name)


@lru_cache
def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(
        index=ensure_index(),
        embedding=get_embeddings(),
        namespace=get_settings().pinecone_namespace,
    )


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": get_settings().top_k})


def _chunk_id(chunk: Document) -> str:
    # Same source + page + position -> same id, so re-ingesting overwrites instead of duplicating.
    # Page matters because PDFs load one Document per page and start_index restarts at 0 on each.
    meta = chunk.metadata
    key = f"{meta.get('source')}:{meta.get('page', '')}:{meta.get('start_index')}"
    return hashlib.sha256(key.encode()).hexdigest()


def add_documents(chunks: list[Document]) -> list[str]:
    ids = [_chunk_id(c) for c in chunks]
    return get_vectorstore().add_documents(chunks, ids=ids)
