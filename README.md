# Documind AI — RAG Document Question Answering

Documind AI is a document-based question-answering application built with
Python, Streamlit, Google Gemini embeddings, NumPy cosine similarity search,
and Gemini answer generation.

Users upload PDF or TXT files, ask questions about their contents, and inspect
retrieved passages that support the generated answer.

## Features

- Multiple PDF and TXT uploads
- Native PDF text extraction with page-level metadata
- OCR fallback for scanned PDF pages using Tesseract
- Overlapping text chunking (800 characters, 150 overlap)
- Gemini embedding generation
- NumPy cosine similarity retrieval
- Gemini grounded answer generation through the Interactions API
- Visible source passages, filenames, page numbers, and similarity scores
- Input validation and error handling
- Uploads up to 100 MB per file and 200 MB combined per processing batch
- Reusable Streamlit-cached Gemini client to avoid closing it between calls

## Project files

```text
Documind-AI/
├── app.py
├── config.py
├── document_loader.py
├── chunker.py
├── embeddings.py
├── retriever.py
├── ai_service.py
├── requirements.txt
├── .env.example
├── .gitignore
├── .streamlit/config.toml
├── sample.txt
└── README.md
```

## Requirements

- Python 3.11 recommended
- Google Gemini API key with access to the configured generation and embedding models
- Tesseract OCR for scanned PDF fallback

### Install Tesseract on macOS

```bash
brew install tesseract
```

If PDFs use languages other than English, install the relevant Tesseract language
data and update `lang="eng"` in `document_loader.py`.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

Set these values in `.env`:

```env
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=gemini-embedding-001
```

The generation model above follows the model name returned by the 404 error in
the development environment. Model availability can vary by API account; if it
is unavailable, use a generation model currently listed for your Gemini API key.
Never commit `.env` or share your API key.

## Run

```bash
python -m streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.
Each PDF or TXT file can be up to 100 MB, and the combined selected files can
total up to 200 MB per processing batch. Restart Streamlit after changing its
upload configuration.

## Test

Upload `sample.txt`, click **Process documents**, and ask:
- What is machine learning?
- What are the types of learning mentioned?
- What is RAG?
- What is the capital of Japan?

The final question should trigger an insufficient-evidence response if the
retrieved context does not contain the answer. Prompt instructions reduce
unsupported answers but cannot guarantee hallucination-free output.

## RAG workflow

1. **Ingestion:** accepts PDF and TXT files.
2. **Extraction:** extracts native PDF text; runs OCR for pages with little/no text.
3. **Chunking:** splits text into chunks of 800 characters with 150 characters of overlap.
4. **Embeddings:** converts document chunks into vectors.
5. **Indexing:** normalizes vectors and holds them in memory.
6. **Retrieval:** embeds the question and ranks chunks using cosine similarity.
7. **Generation:** sends the question and retrieved passages to Gemini.
8. **Grounding:** asks the model to answer from evidence and cite source IDs.
9. **Presentation:** displays the answer and retrieved source passages.

## Design decisions and limitations

- **Chunking:** 800-character chunks and 150-character overlap are baseline values.
  Tune them using representative documents and questions.
- **Vector search:** NumPy performs exact similarity search. It is suitable for a
  small demonstration, but large corpora should use a vector index/database.
- **Memory:** the index is in Streamlit session state and is not persistent across
  a full session reset.
- **OCR:** OCR can misread small print, formulas, and tables. Scanned PDFs may
  require other language data. Large scanned PDFs can take longer to process.
- **Relevance:** this baseline ranks by similarity but does not yet enforce a
  calibrated relevance threshold or rerank results.
- **Grounding:** the prompt instructs the model to abstain when evidence is
  insufficient, but output and citations are not independently verified.
- **Security:** uploaded documents are treated as untrusted content. Add auth,
  access control, retention policies, rate limiting, and monitoring before
  production use.

## Troubleshooting

### `Cannot send a request, as the client has been closed`
This project uses `@st.cache_resource` in `config.py` to reuse one client.
Do not call `client.close()` after an individual request. Stop Streamlit with
Ctrl+C and restart it after code or environment changes.

### `No readable text was found`
Install Tesseract (`brew install tesseract`). OCR fallback is intended for
scanned/image-only pages. If native extraction works poorly for a digital PDF,
inspect the file and extraction output.

### Model `404 NOT_FOUND`
Verify that `GEMINI_MODEL` is available to your API account and that the
installed `google-genai` SDK supports the API used. Model availability can
change over time.

### Missing API key
Check that `.env` is in the project root and contains `GEMINI_API_KEY`.
Restart Streamlit after editing `.env`.

## Future improvements

- Persistent FAISS or Chroma index
- Semantic/keyword hybrid retrieval
- Reranking and relevance thresholds
- Citation support checks
- OCR language selection and PDF page limits
- Automated tests and a retrieval evaluation dataset
- Persistent document library and user-level access controls

## License

Add a license before publishing if you intend to distribute the project.
