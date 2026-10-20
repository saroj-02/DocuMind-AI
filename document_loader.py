from pathlib import Path
import io

import pymupdf as fitz
from PIL import Image
import pytesseract

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_MB = 100
MAX_BATCH_SIZE_MB = 200
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_BATCH_SIZE = MAX_BATCH_SIZE_MB * 1024 * 1024


def validate_upload_batch(uploaded_files):
    oversized_files = [
        uploaded_file.name
        for uploaded_file in uploaded_files
        if uploaded_file.size > MAX_FILE_SIZE
    ]
    if oversized_files:
        names = ", ".join(oversized_files)
        raise ValueError(
            f"Each file must be no larger than {MAX_FILE_SIZE_MB} MB. "
            f"Too large: {names}"
        )

    total_size = sum(uploaded_file.size for uploaded_file in uploaded_files)
    if total_size > MAX_BATCH_SIZE:
        raise ValueError(
            f"The combined upload size must not exceed {MAX_BATCH_SIZE_MB} MB."
        )


def extract_document(uploaded_file):
    filename = Path(uploaded_file.name).name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF and TXT files are supported.")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("The file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            f"The file exceeds the {MAX_FILE_SIZE_MB} MB size limit."
        )

    segments = []

    if extension == ".pdf":
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise ValueError("The PDF is password-protected.")

                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()

                    # OCR only pages that appear to have no usable native text.
                    if len(text) < 20:
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                            image = Image.open(io.BytesIO(pix.tobytes("png")))
                            text = pytesseract.image_to_string(
                                image, lang="eng"
                            ).strip()
                        except pytesseract.TesseractNotFoundError as exc:
                            raise RuntimeError(
                                "Tesseract OCR is not installed or is not on PATH. "
                                "Install it with `brew install tesseract`."
                            ) from exc

                    if text:
                        segments.append({
                            "text": text,
                            "source": filename,
                            "page": page_number,
                        })
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "Could not read this PDF. It may be damaged or encrypted."
            ) from exc
    else:
        try:
            text = file_bytes.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("TXT file must use UTF-8 encoding.") from exc

        if text:
            segments.append({"text": text, "source": filename, "page": None})

    if not segments:
        raise ValueError(
            "No readable text was found. The file may be blank, damaged, "
            "or may require OCR language settings beyond English."
        )

    return segments
