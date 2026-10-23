import os
import secrets
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from app.generation import answer_question
from app.config import chunk_overlap, chunk_size, top_k
from app.chunking import CHUNKING_VERSION
from app.ingestion import MAX_PDF_BYTES, content_identity, prepare_document
from app.retrieval import SecureIndex
from app.security import safe_filename, validate_question

@asynccontextmanager
async def lifespan(application):
    validate_runtime_configuration()
    seed_handbook()
    yield


app = FastAPI(title="RAGShield API", description="PDF question answering with source citations.",
              lifespan=lifespan)
index = SecureIndex()
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def seed_handbook():
    sample = DATA_DIR / "sample_docs" / "Employee-Handbook.pdf"
    if sample.exists():
        try:
            prepared = prepare_document(sample, "local-owner")
            existing = index.get_document("local-owner", prepared["doc_id"])
            settings_changed = existing and (
                existing.get("chunk_size") != prepared["chunk_size"] or
                existing.get("chunk_overlap") != prepared["chunk_overlap"] or
                existing.get("chunking_version") != CHUNKING_VERSION)
            if settings_changed:
                index.delete_document("local-owner", prepared["doc_id"])
            if not existing or settings_changed:
                index.upsert(prepared["chunks"])
        except (OSError, ValueError) as exc:
            logging.error("Could not index sample handbook (error type: %s).", type(exc).__name__)


def validate_runtime_configuration():
    token = os.getenv("RAGSHIELD_ACCESS_TOKEN", "")
    if len(token) < 32 or token == "replace-with-a-long-random-token":
        raise RuntimeError("Set RAGSHIELD_ACCESS_TOKEN to a private random value of at least 32 characters before startup.")
    # Validate configuration at startup, before serving requests.
    chunk_size()
    chunk_overlap()
    top_k()
    from app.config import max_distance, max_extracted_chars, max_pdf_pages, max_pdf_size_mb
    max_distance()
    max_pdf_size_mb()
    max_pdf_pages()
    max_extracted_chars()


def is_read_only_demo() -> bool:
    return os.getenv("RAGSHIELD_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}


def authenticate(authorization: str | None) -> str:
    expected = os.getenv("RAGSHIELD_ACCESS_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Set RAGSHIELD_ACCESS_TOKEN in .env before starting the API.")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, "Invalid bearer token.")
    return "local-owner"


def remove_legacy_upload(owner_id: str, doc_id: str, filename: str) -> None:
    """Remove matching files left by older builds after verifying their content ID."""
    expected_suffix = "_" + safe_filename(filename)
    if not UPLOAD_DIR.exists():
        return
    for candidate in UPLOAD_DIR.iterdir():
        prefix, separator, suffix = candidate.name.partition("_")
        if (not separator or len(prefix) != 12 or
                any(char not in "0123456789abcdef" for char in prefix.lower()) or
                "_" + suffix != expected_suffix):
            continue
        try:
            if not candidate.is_file() or candidate.stat().st_size > MAX_PDF_BYTES:
                continue
            if content_identity(candidate.read_bytes(), owner_id)[1] == doc_id:
                candidate.unlink(missing_ok=True)
        except OSError as exc:
            logging.warning("Could not remove a legacy upload (error type: %s).", type(exc).__name__)


@app.get("/health")
def health():
    return {"status": "ok", "security": "document text is untrusted; injection chunks are filtered"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    owner = authenticate(authorization)
    if is_read_only_demo():
        raise HTTPException(403, "Document uploads are disabled in read-only demo mode.")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are accepted.")
    payload = await file.read(MAX_PDF_BYTES + 1)
    if len(payload) > MAX_PDF_BYTES:
        raise HTTPException(413, f"PDF must be no larger than {MAX_PDF_BYTES // (1024 * 1024)} MB.")
    safe_name = safe_filename(file.filename)
    _, doc_id = content_identity(payload, owner)
    existing = index.get_document(owner, doc_id)
    if existing:
        same_chunk_settings = (existing.get("chunk_size") == chunk_size() and
                               existing.get("chunk_overlap") == chunk_overlap() and
                               existing.get("chunking_version") == CHUNKING_VERSION)
        if same_chunk_settings:
            return {"doc_id": doc_id, "filename": existing["filename"], "pages": None,
                    "flagged_chunks": 0, "redacted_secret_chunks": 0, "already_indexed": True}
        metadata = index.get_document(owner, doc_id)
        index.delete_document(owner, doc_id)
        if metadata and metadata.get("filename"):
            remove_legacy_upload(owner, doc_id, metadata["filename"])

    # The stored filename is generated by the server; the upload name is display metadata only.
    path = UPLOAD_DIR / f"{secrets.token_hex(16)}.pdf"
    try:
        path.write_bytes(payload)
        result = prepare_document(path, owner, display_name=safe_name)
        index.upsert(result["chunks"])
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        path.unlink(missing_ok=True)
        logging.error("Document ingestion failed (error type: %s).", type(exc).__name__)
        raise HTTPException(500, "Document ingestion failed. Check the API log for details.") from exc
    finally:
        path.unlink(missing_ok=True)
    return {k: result[k] for k in ("doc_id", "filename", "pages", "flagged_chunks", "redacted_secret_chunks")} | {"already_indexed": False}


@app.get("/documents")
def list_documents(authorization: str | None = Header(default=None)):
    owner = authenticate(authorization)
    try:
        return [{"doc_id": key, "filename": value} for key, value in index.documents(owner).items()]
    except Exception as exc:
        logging.error("Document listing failed (error type: %s).", type(exc).__name__)
        raise HTTPException(503, "Document listing is temporarily unavailable.") from exc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, authorization: str | None = Header(default=None)):
    owner = authenticate(authorization)
    if is_read_only_demo():
        raise HTTPException(403, "Document deletion is disabled in read-only demo mode.")
    try:
        # Owner-scoped existence check prevents ID tampering and hides other owners' IDs.
        metadata = index.get_document(owner, doc_id)
        if not metadata:
            raise HTTPException(404, "Document not found.")
        index.delete_document(owner, doc_id)
        if metadata.get("filename"):
            remove_legacy_upload(owner, doc_id, metadata["filename"])
    except HTTPException:
        raise
    except Exception as exc:
        logging.error("Document deletion failed (error type: %s).", type(exc).__name__)
        raise HTTPException(503, "Document deletion is temporarily unavailable.") from exc
    return {"doc_id": doc_id, "deleted": True}


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    doc_ids: list[str] = Field(min_length=1, max_length=50)
    k: int = Field(default_factory=top_k, ge=1, le=20)
    strategy: str = Field(default="semantic", pattern="^(semantic|keyword|hybrid)$")


@app.post("/query")
def query(body: Query, authorization: str | None = Header(default=None)):
    owner = authenticate(authorization)
    try:
        question = validate_question(body.question)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    allowed = set(index.documents(owner))
    if not set(body.doc_ids).issubset(allowed):
        raise HTTPException(403, "One or more selected documents are not available to this user.")
    try:
        sources = index.search(question, owner, body.doc_ids, k=body.k, strategy=body.strategy)
    except Exception as exc:
        logging.error("Retrieval failed (error type: %s).", type(exc).__name__)
        raise HTTPException(503, "Document search is temporarily unavailable.") from exc
    for position, source in enumerate(sources, start=1):
        source["source_id"] = f"S{position}"
    generated = answer_question(question, sources)
    def public_source(source):
        return {"source_id": source["source_id"], "filename": source["filename"],
                "doc_id": source["doc_id"], "chunk_id": source["chunk_id"],
                "page": source["page"], "text": source["text"],
                "distance": round(source["distance"], 3) if source["distance"] is not None else None,
                "cosine_similarity": round(source["cosine_similarity"], 3) if source["cosine_similarity"] is not None else None,
                "retrieval_score": round(source["retrieval_score"], 5),
                "retrieval_method": source["retrieval_method"]}
    return {
        "answer": generated["answer"],
        "sources": [public_source(s) for s in sources],
        "cited_sources": [public_source(s) for s in generated["cited_sources"]],
        "provider": generated["provider"],
        "tokens": generated["tokens"],
        "warning": generated["warning"],
    }
