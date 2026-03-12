"""Generate embeddings for synthetic candidates and compute all pairwise similarities.

Usage:
    uv run python scripts/similarity_experiment.py

Reads  : synthetic_candidates.json
Writes : similarity_results.json
"""

import asyncio
import itertools
import json
from pathlib import Path

import numpy as np
from pydantic_ai import Embedder
from pydantic_ai.embeddings.google import GoogleEmbeddingModel
from pydantic_ai.providers.google import GoogleProvider

from idea_pipeline.core.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "synthetic_candidates.json"
OUTPUT_FILE = PROJECT_ROOT / "similarity_results.json"


def _normalize(vectors: list[list[float]]) -> np.ndarray:
    arr = np.array(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return arr / norms


async def main() -> None:
    # Load synthetic candidates
    candidates = json.loads(INPUT_FILE.read_text())

    # Build embedding texts the same way production does
    texts = [f"{c['one_liner']} {c['solution_overview']}" for c in candidates]

    # Embed using the same model and settings as production
    provider = GoogleProvider(api_key=settings.gemini_api_key)
    model = GoogleEmbeddingModel("gemini-embedding-001", provider=provider)
    embedder = Embedder(
        model,
        settings={"google_task_type": "SEMANTIC_SIMILARITY", "dimensions": 768},
    )

    print(f"Embedding {len(texts)} candidates …")
    result = await embedder.embed(texts, input_type="document")
    normalized = _normalize(result.embeddings)
    print("Done.\n")

    # Compute all pairwise cosine similarities (dot product of normalized vecs)
    similarity_matrix = (normalized @ normalized.T).tolist()

    # Build structured output
    id_list = [c["id"] for c in candidates]
    label_map = {c["id"]: c["label"] for c in candidates}
    relation_map = {c["id"]: c["relation"] for c in candidates}
    group_map = {c["id"]: c["group"] for c in candidates}

    pairs = []
    for i, j in itertools.combinations(range(len(candidates)), 2):
        pairs.append(
            {
                "a": id_list[i],
                "a_label": label_map[id_list[i]],
                "b": id_list[j],
                "b_label": label_map[id_list[j]],
                "same_group": group_map[id_list[i]] == group_map[id_list[j]],
                "relation_a": relation_map[id_list[i]],
                "relation_b": relation_map[id_list[j]],
                "similarity": round(similarity_matrix[i][j], 6),
            }
        )

    # Sort by similarity descending for easy inspection
    pairs.sort(key=lambda p: p["similarity"], reverse=True)

    output = {
        "candidates": [
            {"id": c["id"], "label": c["label"], "group": c["group"], "relation": c["relation"]}
            for c in candidates
        ],
        "similarity_matrix": {
            id_list[i]: {id_list[j]: round(similarity_matrix[i][j], 6) for j in range(len(id_list))}
            for i in range(len(id_list))
        },
        "pairs_ranked": pairs,
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"Results written to {OUTPUT_FILE}\n")

    # Print a summary table
    print(f"{'Pair':<90} {'Sim':>8}  Relation")
    print("-" * 115)
    for p in pairs:
        a_short = p["a"]
        b_short = p["b"]
        pair_label = f"{a_short} <-> {b_short}"
        relation = f"{p['relation_a']} / {p['relation_b']}"
        same = " (same group)" if p["same_group"] else ""
        print(f"{pair_label:<90} {p['similarity']:>8.4f}  {relation}{same}")


if __name__ == "__main__":
    asyncio.run(main())
