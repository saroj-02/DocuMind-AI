import numpy as np
from google.genai import types

from config import get_client, EMBEDDING_MODEL


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    if not texts:
        return []

    client = get_client()
    vectors = []

    # Batch requests to avoid sending an entire large document in one request.
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )

        if not response.embeddings or len(response.embeddings) != len(batch):
            raise RuntimeError("Embedding API returned an unexpected response.")

        for item in response.embeddings:
            vector = np.asarray(item.values, dtype=np.float32)
            if vector.size == 0 or not np.all(np.isfinite(vector)):
                raise RuntimeError("Embedding API returned an invalid vector.")
            vectors.append(vector)

    return vectors


def embed_query(question):
    return embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
