import uuid
from collections.abc import Sequence

import chromadb
import numpy as np
from pydantic_ai import Embedder
from pydantic_ai.embeddings.google import GoogleEmbeddingModel
from pydantic_ai.providers.google import GoogleProvider

from idea_pipeline.core.settings import settings

_COLLECTION_NAME = "candidates"


def _create_embedder() -> Embedder:
    provider = GoogleProvider(api_key=settings.gemini_api_key)
    model = GoogleEmbeddingModel("gemini-embedding-001", provider=provider)
    return Embedder(
        model,
        settings={"google_task_type": "SEMANTIC_SIMILARITY", "dimensions": 768},
    )


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _normalize(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    arr = np.array(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = arr / norms
    return normalized.tolist()


async def store_candidate_embeddings(
    candidate_ids: list[uuid.UUID],
    texts: list[str],
) -> None:
    embedder = _create_embedder()
    result = await embedder.embed(texts, input_type="document")
    normalized = _normalize(result.embeddings)

    collection = _get_collection()
    collection.upsert(
        ids=[str(cid) for cid in candidate_ids],
        embeddings=normalized,
        documents=texts,
    )


async def query_similar_candidates(
    text: str,
    n_results: int = 5,
) -> list[tuple[uuid.UUID, float]]:
    embedder = _create_embedder()
    result = await embedder.embed(text, input_type="query")
    normalized = _normalize(result.embeddings)

    collection = _get_collection()
    results = collection.query(
        query_embeddings=normalized,
        n_results=n_results,
    )

    pairs: list[tuple[uuid.UUID, float]] = []
    if results["ids"] and results["distances"]:
        for cid, distance in zip(results["ids"][0], results["distances"][0]):
            pairs.append((uuid.UUID(cid), distance))
    return pairs


def get_max_similarity_to_published(
    candidate_ids: list[uuid.UUID],
    published_ids: list[uuid.UUID],
) -> dict[uuid.UUID, float]:
    """For each candidate, find max cosine similarity to any published candidate.

    Returns a dict mapping candidate_id -> max similarity (0.0 if no published
    embeddings exist or the candidate has no embedding).
    """
    if not candidate_ids or not published_ids:
        return {}

    collection = _get_collection()

    str_candidate_ids = [str(cid) for cid in candidate_ids]
    str_published_ids = [str(pid) for pid in published_ids]

    candidate_result = collection.get(
        ids=str_candidate_ids, include=["embeddings"]
    )
    published_result = collection.get(
        ids=str_published_ids, include=["embeddings"]
    )

    if (
        not candidate_result["embeddings"]
        or not published_result["embeddings"]
    ):
        return {}

    candidate_vecs = np.array(candidate_result["embeddings"], dtype=np.float64)
    published_vecs = np.array(published_result["embeddings"], dtype=np.float64)

    # Embeddings are already normalized, so dot product = cosine similarity
    similarity_matrix = candidate_vecs @ published_vecs.T
    max_similarities = similarity_matrix.max(axis=1)

    result: dict[uuid.UUID, float] = {}
    for str_id, max_sim in zip(candidate_result["ids"], max_similarities):
        result[uuid.UUID(str_id)] = float(max_sim)

    return result
