from pathlib import Path

import pytest

from app.ingestion import content_identity, extract_pdf, extract_pdf_bytes, prepare_document

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_docs" / "Employee-Handbook.pdf"


def test_sample_pdf_extracts_page_numbered_text():
    pages = extract_pdf(SAMPLE)
    assert pages
    assert all(page["page"] > 0 and page["text"].strip() for page in pages)
    assert any("sick leave" in page["text"].lower() for page in pages)


def test_non_pdf_bytes_are_rejected():
    with pytest.raises(ValueError, match="not a valid PDF"):
        extract_pdf_bytes(b"not a pdf")


def test_empty_pdf_is_rejected():
    import pymupdf
    empty = pymupdf.open()
    empty.new_page()
    data = empty.tobytes()
    empty.close()
    with pytest.raises(ValueError, match="No selectable text"):
        extract_pdf_bytes(data)


def test_malformed_pdf_with_signature_is_rejected_safely():
    with pytest.raises(ValueError, match="valid, readable PDF"):
        extract_pdf_bytes(b"%PDF-1.7 this is not a valid PDF body")


def test_configured_upload_size_limit_is_enforced(monkeypatch):
    import app.ingestion as ingestion

    monkeypatch.setattr(ingestion, "MAX_PDF_BYTES", 8)
    with pytest.raises(ValueError, match="no larger than"):
        extract_pdf_bytes(b"%PDF-123456789")


def test_extracted_text_limit_is_enforced(monkeypatch):
    import pymupdf
    import app.ingestion as ingestion

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This policy text is longer than the test limit.")
    data = doc.tobytes()
    doc.close()
    monkeypatch.setattr(ingestion, "MAX_EXTRACTED_CHARS", 12)
    with pytest.raises(ValueError, match="too much extractable text"):
        extract_pdf_bytes(data)


def test_page_count_limit_is_enforced(monkeypatch):
    import pymupdf
    import app.ingestion as ingestion

    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    monkeypatch.setattr(ingestion, "MAX_PDF_PAGES", 1)
    with pytest.raises(ValueError, match="limited to 1 pages"):
        extract_pdf_bytes(data)


def test_encrypted_pdf_is_rejected():
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Private policy")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                       owner_pw="test-owner", user_pw="test-user")
    doc.close()
    with pytest.raises(ValueError, match="Password-protected"):
        extract_pdf_bytes(data)


def test_content_identity_is_stable_and_owner_scoped():
    data = SAMPLE.read_bytes()
    first_hash, first_id = content_identity(data, "owner-a")
    second_hash, second_id = content_identity(data, "owner-a")
    other_hash, other_id = content_identity(data, "owner-b")
    assert first_hash == second_hash == other_hash
    assert first_id == second_id
    assert first_id != other_id


def test_prepare_document_has_stable_document_and_chunk_metadata():
    first = prepare_document(SAMPLE, "test-owner", size=100, overlap=20)
    second = prepare_document(SAMPLE, "test-owner", size=100, overlap=20)
    assert first["doc_id"] == second["doc_id"]
    assert first["filename"] == SAMPLE.name
    assert first["chunks"]
    assert all(chunk["doc_id"] == first["doc_id"] for chunk in first["chunks"])
    assert all(chunk["chunk_id"] == chunk["id"] for chunk in first["chunks"])
    assert all(chunk["chunking_version"] == first["chunking_version"]
               for chunk in first["chunks"])
