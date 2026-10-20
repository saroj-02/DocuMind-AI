import streamlit as st

from document_loader import (
    MAX_BATCH_SIZE_MB,
    MAX_FILE_SIZE_MB,
    extract_document,
    validate_upload_batch,
)
from chunker import create_chunks
from retriever import build_index, retrieve
from ai_service import generate_answer

st.set_page_config(
    page_title="Documind AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Documind AI")
st.caption("Document-based question answering powered by retrieval-augmented generation (RAG).")
st.info(
    "Upload PDF or TXT documents, process them, then ask questions. "
    "Answers include retrieved passages so you can verify the evidence."
)

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None
if "document_names" not in st.session_state:
    st.session_state.document_names = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Document library")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help=(
            f"Maximum {MAX_FILE_SIZE_MB} MB per file and "
            f"{MAX_BATCH_SIZE_MB} MB combined."
        ),
    )

    if st.button("Process documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one document first.")
        else:
            try:
                validate_upload_batch(uploaded_files)
            except ValueError as exc:
                st.error(str(exc))
            else:
                segments = []
                accepted_names = []
                errors = []

                with st.spinner("Extracting document text..."):
                    for uploaded_file in uploaded_files:
                        try:
                            extracted = extract_document(uploaded_file)
                            segments.extend(extracted)
                            accepted_names.append(uploaded_file.name)
                        except Exception as exc:
                            errors.append(f"{uploaded_file.name}: {exc}")

                for error in errors:
                    st.warning(error)

                if segments:
                    try:
                        with st.spinner("Chunking documents and creating embeddings..."):
                            chunks = create_chunks(segments, chunk_size=800, overlap=150)
                            new_index = build_index(chunks)

                        # Only replace the current index after the new index succeeds.
                        st.session_state.rag_index = new_index
                        st.session_state.document_names = accepted_names
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.messages = []
                        st.success(
                            f"Processed {len(accepted_names)} document(s) into {len(chunks)} chunks."
                        )
                    except Exception as exc:
                        st.error(f"Could not analyze the documents: {exc}")
                elif not errors:
                    st.warning("No readable text was found in the uploaded documents.")

    st.divider()
    st.subheader("Current index")
    if st.session_state.rag_index is not None:
        st.write(f"**Documents:** {len(st.session_state.document_names)}")
        st.write(f"**Chunks:** {st.session_state.chunk_count}")
        for name in st.session_state.document_names:
            st.write(f"📄 {name}")
        if st.button("Clear document index"):
            st.session_state.rag_index = None
            st.session_state.document_names = []
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("No documents indexed yet.")

st.subheader("Ask a question")
with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "Your question",
        placeholder="What are the main topics in these documents?",
    )
    submitted = st.form_submit_button("Ask Documind AI", type="primary")

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    elif st.session_state.rag_index is None:
        st.warning("Upload and process documents first.")
    else:
        try:
            with st.spinner("Retrieving evidence and generating an answer..."):
                sources = retrieve(st.session_state.rag_index, question.strip())
                answer = generate_answer(question.strip(), sources)
            st.session_state.messages.append({
                "question": question.strip(),
                "answer": answer,
                "sources": sources,
            })
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")

for message in reversed(st.session_state.messages):
    st.divider()
    st.markdown("#### Question")
    st.write(message["question"])
    st.markdown("#### Answer")
    st.markdown(message["answer"])

    with st.expander(f"View retrieved source passages ({len(message['sources'])})", expanded=True):
        for number, source in enumerate(message["sources"], start=1):
            page = source.get("page")
            page_label = str(page) if page is not None else "N/A"
            st.markdown(f"**[S{number}] {source['source']}** — Page {page_label}")
            st.caption(f"Cosine similarity: {source['score']:.3f}")
            st.write(source["text"])
            st.divider()
