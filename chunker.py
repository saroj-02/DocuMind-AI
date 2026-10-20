def split_text(text, chunk_size=800, overlap=150):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap

    return chunks


def create_chunks(segments, chunk_size=800, overlap=150):
    documents = []
    for segment in segments:
        for index, piece in enumerate(
            split_text(segment["text"], chunk_size, overlap), start=1
        ):
            documents.append({
                "text": piece,
                "source": segment["source"],
                "page": segment.get("page"),
                "chunk_number": index,
            })
    return documents
