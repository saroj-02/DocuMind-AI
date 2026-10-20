import unittest
from unittest.mock import patch

import pymupdf

from chunker import split_text
from document_loader import (
    MAX_BATCH_SIZE,
    MAX_FILE_SIZE,
    extract_document,
    validate_upload_batch,
)


class FakeUploadedFile:
    def __init__(self, name, content=b"", size=None):
        self.name = name
        self.content = content
        self.size = len(content) if size is None else size

    def getvalue(self):
        return self.content


class DocumentLoaderTests(unittest.TestCase):
    def test_validates_file_and_batch_size_limits(self):
        validate_upload_batch([
            FakeUploadedFile("first.pdf", size=MAX_FILE_SIZE),
            FakeUploadedFile("second.txt", size=MAX_FILE_SIZE),
        ])

        with self.assertRaisesRegex(ValueError, "100 MB"):
            validate_upload_batch([
                FakeUploadedFile("large.pdf", size=MAX_FILE_SIZE + 1),
            ])

        with self.assertRaisesRegex(ValueError, "200 MB"):
            validate_upload_batch([
                FakeUploadedFile("first.pdf", size=MAX_BATCH_SIZE // 2),
                FakeUploadedFile("second.pdf", size=MAX_BATCH_SIZE // 2),
                FakeUploadedFile("third.txt", size=1),
            ])

    def test_rejects_file_over_extraction_limit(self):
        with patch("document_loader.MAX_FILE_SIZE", 3):
            with self.assertRaisesRegex(ValueError, "100 MB"):
                extract_document(FakeUploadedFile("large.txt", b"four"))

    def test_extracts_utf8_text_file(self):
        segments = extract_document(
            FakeUploadedFile("notes.txt", b"\xef\xbb\xbfStudy material")
        )

        self.assertEqual(
            segments,
            [{"text": "Study material", "source": "notes.txt", "page": None}],
        )

    def test_extracts_native_pdf_text_with_page_number(self):
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "PDF study material")
        content = pdf.tobytes()
        pdf.close()

        segments = extract_document(FakeUploadedFile("notes.pdf", content))

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["source"], "notes.pdf")
        self.assertEqual(segments[0]["page"], 1)
        self.assertIn("PDF study material", segments[0]["text"])

    def test_chunking_still_produces_overlapping_chunks(self):
        chunks = split_text("abcdefghij", chunk_size=6, overlap=2)

        self.assertEqual(chunks, ["abcdef", "efghij"])


if __name__ == "__main__":
    unittest.main()
