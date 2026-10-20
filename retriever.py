import numpy as np

from embeddings import embed_texts, embed_query


def build_index(chunks):
    if not chunks:
        raise ValueError("No document chunks were created.")

    vectors = embed_texts(
        [chunk["text"] for chunk in chunks],
        task_type="RETRIEVAL_DOCUMENT",
    )
    matrix = np.vstack(vectors)

    if not np.all(np.isfinite(matrix)):
        raise ValueError("The document index contains invalid vectors.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A document embedding has zero magnitude.")

    return {"chunks": chunks, "vectors": matrix / norms}


def retrieve(index, question, top_k=4):
    if not question or not question.strip():
        raise ValueError("Please enter a question.")
    if not index or not index.get("chunks"):
        raise ValueError("Process documents before asking questions.")

    query_vector = embed_query(question)
    norm = np.linalg.norm(query_vector)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("The question embedding is invalid.")

    query_vector = query_vector / norm
    scores = index["vectors"] @ query_vector
    best_indices = np.argsort(scores)[::-1][:max(1, int(top_k))]

    results = []
    for idx in best_indices:
        item = dict(index["chunks"][int(idx)])
        item["score"] = float(scores[idx])
        results.append(item)
    return results
