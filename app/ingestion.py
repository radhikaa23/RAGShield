"""PDF validation, page-preserving text extraction, and document chunk metadata."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf

from app.chunking import CHUNKING_VERSION, chunk_pages
from app.config import chunk_overlap, chunk_size, max_extracted_chars, max_pdf_pages, max_pdf_size_mb
from app.security import inspect_text, redact_secrets

MAX_PDF_BYTES = max_pdf_size_mb() * 1024 * 1024
MAX_PDF_PAGES = max_pdf_pages()
MAX_EXTRACTED_CHARS = max_extracted_chars()


def content_identity(data: bytes, owner_id: str) -> tuple[str, str]:
    """Return a content hash and a stable owner-scoped document ID."""
    content_hash = hashlib.sha256(data).hexdigest()
    doc_id = hashlib.sha256(f"{owner_id}\0{content_hash}".encode("utf-8")).hexdigest()[:24]
    return content_hash, doc_id


def extract_pdf_bytes(data: bytes) -> tuple[list[dict], int]:
    if not data.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"PDFs must be no larger than {MAX_PDF_BYTES // (1024 * 1024)} MB.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")
            if pdf.page_count > MAX_PDF_PAGES:
                raise ValueError(f"PDFs are limited to {MAX_PDF_PAGES} pages.")
            page_count = pdf.page_count
            pages = []
            total_text_chars = 0
            for page_number, page in enumerate(pdf, start=1):
                raw_text = page.get_text("text").strip()
                if raw_text:
                    total_text_chars += len(raw_text)
                    if total_text_chars > MAX_EXTRACTED_CHARS:
                        raise ValueError("The PDF contains too much extractable text for this service.")
                    inspection = inspect_text(raw_text)
                    pages.append({
                        "page": page_number,
                        "text": redact_secrets(raw_text),
                        "prompt_injection": inspection["prompt_injection"],
                        "secret_detected": inspection["secret_detected"],
                    })
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The PDF could not be opened. Check that it is a valid, readable PDF.") from exc
    if not pages:
        raise ValueError("No selectable text was found. Scanned PDFs need OCR, which this project does not include.")
    return pages, page_count


def extract_pdf(path: Path) -> list[dict]:
    """Extract each non-empty page from a PDF path, keeping original page numbers."""
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are accepted.")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError("The uploaded PDF could not be read.") from exc
    return extract_pdf_bytes(data)[0]


def prepare_document(
    path: Path,
    owner_id: str,
    display_name: str | None = None,
    size: int | None = None,
    overlap: int | None = None,
) -> dict:
    data = path.read_bytes()
    pages, physical_page_count = extract_pdf_bytes(data)
    from app.security import safe_filename

    filename = safe_filename(display_name or path.name)
    content_hash, doc_id = content_identity(data, owner_id)
    effective_size = size if size is not None else chunk_size()
    effective_overlap = overlap if overlap is not None else chunk_overlap()
    chunks = chunk_pages(pages, size=effective_size, overlap=effective_overlap)
    records = []
    flagged = 0
    redacted_secret_chunks = 0
    page_flags = {page["page"]: page for page in pages}
    for chunk in chunks:
        risk = inspect_text(chunk.text)
        page_flag = page_flags[chunk.page]
        injection_flag = risk["prompt_injection"] or page_flag["prompt_injection"]
        secret_redacted = page_flag["secret_detected"]
        flagged += int(injection_flag)
        redacted_secret_chunks += int(secret_redacted)
        records.append({
            "id": f"{doc_id}:{chunk.index}",
            "chunk_id": f"{doc_id}:{chunk.index}",
            "text": chunk.text,
            "page": chunk.page,
            "doc_id": doc_id,
            "filename": filename,
            "owner_id": owner_id,
            "content_hash": content_hash,
            "chunk_size": effective_size,
            "chunk_overlap": effective_overlap,
            "chunking_version": CHUNKING_VERSION,
            "injection_flag": injection_flag,
            "secret_redacted": secret_redacted,
        })
    return {
        "doc_id": doc_id,
        "content_hash": content_hash,
        "chunk_size": effective_size,
        "chunk_overlap": effective_overlap,
        "chunking_version": CHUNKING_VERSION,
        "filename": filename,
        "pages": physical_page_count,
        "chunks": records,
        "flagged_chunks": flagged,
        "redacted_secret_chunks": redacted_secret_chunks,
    }
