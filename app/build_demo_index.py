"""Create the read-only demo's Chroma index during the hosting build."""
from __future__ import annotations

import logging
import os

from app.chunking import CHUNKING_VERSION
from app.config import PROJECT_ROOT, chunk_overlap, chunk_size
from app.ingestion import prepare_document
from app.retrieval import SecureIndex


def main() -> None:
    # Keep these build-time settings aligned with render.yaml. setdefault permits
    # an explicit environment override when building outside Render.
    os.environ.setdefault("RAGSHIELD_EMBEDDING_BACKEND", "chroma_onnx")
    os.environ.setdefault("RAGSHIELD_CHUNK_SIZE", "250")
    os.environ.setdefault("RAGSHIELD_CHUNK_OVERLAP", "38")

    sample = PROJECT_ROOT / "data" / "sample_docs" / "Employee-Handbook.pdf"
    if not sample.is_file():
        raise FileNotFoundError(f"Required demo PDF was not found: {sample.name}")

    index = SecureIndex()
    prepared = prepare_document(sample, "local-owner")
    existing = index.get_document("local-owner", prepared["doc_id"])
    settings_changed = existing and (
        existing.get("chunk_size") != prepared["chunk_size"]
        or existing.get("chunk_overlap") != prepared["chunk_overlap"]
        or existing.get("chunking_version") != CHUNKING_VERSION
    )
    if settings_changed:
        index.delete_document("local-owner", prepared["doc_id"])
    if not existing or settings_changed:
        index.upsert(prepared["chunks"])

    logging.info("Built demo index: %s pages, %s chunks.", prepared["pages"], len(prepared["chunks"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
