from app.config import PROJECT_ROOT, max_distance, top_k
from app.chunking import CHUNKING_VERSION, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL
import re
import math
import os

EMBEDDING_BATCH_SIZE = 16


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./-]+", text.lower())


def _bm25(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Small BM25 scorer; keeps keyword retrieval dependency-free and inspectable."""
    tokenized = [_terms(text) for text in documents]
    if not tokenized:
        return []
    avg_len = sum(map(len, tokenized)) / len(tokenized) or 1
    qterms = set(_terms(query))
    scores = [0.0] * len(tokenized)
    for term in qterms:
        df = sum(term in set(doc) for doc in tokenized)
        if not df:
            continue
        idf = math.log(1 + (len(tokenized) - df + 0.5) / (df + 0.5))
        for i, doc in enumerate(tokenized):
            tf = doc.count(term)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg_len))
    return scores


class SecureIndex:
    def __init__(self, directory: str | None = None):
        # Chroma (and ONNX Runtime in the demo backend) is heavy to import.
        # Import it only when an API request actually needs the index.
        from chromadb import EphemeralClient, PersistentClient

        self.client = EphemeralClient() if directory == ":memory:" else PersistentClient(
            path=directory or str(PROJECT_ROOT / "data" / "chroma"))
        backend = os.getenv("RAGSHIELD_EMBEDDING_BACKEND", "sentence_transformers").strip().lower()
        if backend == "chroma_onnx":
            # Chroma's default ONNX implementation uses the same all-MiniLM-L6-v2
            # model family without loading PyTorch; intended for low-memory hosting.
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            embed = DefaultEmbeddingFunction()
        elif backend == "sentence_transformers":
            # Import lazily so the low-memory ONNX demo does not initialize PyTorch.
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            embed = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
            embed._model.max_seq_length = EMBEDDING_MAX_TOKENS
        else:
            raise ValueError("RAGSHIELD_EMBEDDING_BACKEND must be sentence_transformers or chroma_onnx.")
        self.collection = self.client.get_or_create_collection(
            "ragshield", embedding_function=embed, metadata={"hnsw:space": "cosine"})

    def upsert(self, records: list[dict]):
        if not records:
            return
        # Keep embedding inference's temporary arrays small on memory-limited hosts.
        metadata_keys = ("page", "doc_id", "chunk_id", "filename", "owner_id", "content_hash",
                         "chunk_size", "chunk_overlap", "chunking_version", "injection_flag",
                         "secret_redacted")
        for offset in range(0, len(records), EMBEDDING_BATCH_SIZE):
            batch = records[offset:offset + EMBEDDING_BATCH_SIZE]
            self.collection.upsert(
                ids=[record["id"] for record in batch],
                documents=[record["text"] for record in batch],
                metadatas=[{key: record[key] for key in metadata_keys} for record in batch],
            )

    def delete_document(self, owner_id: str, doc_id: str):
        """Delete only one owner's chunks for one content-addressed document."""
        self.collection.delete(where={"$and": [{"owner_id": owner_id}, {"doc_id": doc_id}]})

    def get_document(self, owner_id: str, doc_id: str):
        result = self.collection.get(
            where={"$and": [{"owner_id": owner_id}, {"doc_id": doc_id}]},
            include=["metadatas"], limit=1)
        metadatas = result.get("metadatas") or []
        if not metadatas:
            return None
        metadata = metadatas[0]
        if metadata.get("owner_id") != owner_id or metadata.get("doc_id") != doc_id:
            return None
        return metadata

    def has_document(self, owner_id: str, doc_id: str) -> bool:
        return self.get_document(owner_id, doc_id) is not None

    def documents(self, owner_id: str):
        result = self.collection.get(where={"owner_id": owner_id}, include=["metadatas"])
        # Do not rely on the vector-store filter alone for authorization.
        metadatas = [meta for meta in (result.get("metadatas") or [])
                     if meta.get("owner_id") == owner_id and meta.get("doc_id")]
        current_names = {meta.get("filename") for meta in metadatas
                         if meta.get("content_hash")}
        # Older app versions indexed the seeded handbook without a content hash.
        # Hide that stale row when the current hash-backed version is available.
        by_content = {}
        for meta in metadatas:
            if not meta.get("content_hash") and meta.get("filename") in current_names:
                continue
            content_key = meta.get("content_hash") or meta["doc_id"]
            current = by_content.get(content_key)
            if current is None or meta.get("chunking_version", 0) > current.get("chunking_version", 0):
                by_content[content_key] = meta
        return {meta["doc_id"]: meta["filename"] for meta in by_content.values()}

    def search(self, question: str, owner_id: str, doc_ids: list[str], k: int | None = None,
               distance_threshold: float | None = None, strategy: str = "semantic"):
        if not owner_id or not doc_ids or any(not isinstance(doc_id, str) or not doc_id for doc_id in doc_ids):
            return []
        authorized_doc_ids = set(doc_ids)
        k = k or top_k()
        if k < 1:
            raise ValueError("k must be positive")
        if strategy not in {"semantic", "keyword", "hybrid"}:
            raise ValueError("strategy must be semantic, keyword, or hybrid")
        threshold = distance_threshold if distance_threshold is not None else max_distance()
        if not 0 <= threshold <= 2:
            raise ValueError("cosine distance threshold must be between 0 and 2")
        candidate_count = min(max(k * 10, 50), self.collection.count())
        if candidate_count == 0:
            return []
        where = {"$and": [{"owner_id": owner_id}, {"doc_id": {"$in": doc_ids}}]}
        raw_rows = []
        if strategy in {"semantic", "hybrid"}:
            result = self.collection.query(query_texts=[question], n_results=candidate_count,
                where=where, include=["documents", "metadatas", "distances"])
            for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
                if (meta.get("owner_id") != owner_id or meta.get("doc_id") not in authorized_doc_ids
                        or meta.get("injection_flag") or distance > threshold):
                    continue
                raw_rows.append({"text": doc, "page": meta["page"], "filename": meta["filename"],
                    "doc_id": meta["doc_id"], "chunk_id": meta.get("chunk_id", ""),
                    "distance": float(distance), "cosine_similarity": 1 - float(distance),
                    "retrieval_score": 1 - float(distance), "retrieval_method": "semantic"})
        semantic_rows = raw_rows
        if strategy in {"keyword", "hybrid"}:
            all_docs = self.collection.get(where=where, include=["documents", "metadatas"])
            docs = all_docs.get("documents") or []
            metas = all_docs.get("metadatas") or []
            scores = _bm25(question, docs)
            keyword_rows = []
            for doc, meta, score in zip(docs, metas, scores):
                if (meta.get("owner_id") != owner_id or meta.get("doc_id") not in authorized_doc_ids
                        or meta.get("injection_flag") or score <= 0):
                    continue
                keyword_rows.append({"text": doc, "page": meta["page"], "filename": meta["filename"],
                    "doc_id": meta["doc_id"], "chunk_id": meta.get("chunk_id", ""),
                    "distance": None, "cosine_similarity": None, "retrieval_score": float(score),
                    "retrieval_method": "keyword"})
            keyword_rows.sort(key=lambda row: row["retrieval_score"], reverse=True)
            if strategy == "keyword":
                raw_rows = keyword_rows
            else:
                # Reciprocal Rank Fusion over the two independently ranked lists.
                fused = {}
                for ranked in (semantic_rows, keyword_rows):
                    for rank, row in enumerate(ranked, start=1):
                        key = (row["doc_id"], row["chunk_id"])
                        fused.setdefault(key, {**row, "retrieval_score": 0.0, "retrieval_method": "hybrid"})
                        fused[key]["retrieval_score"] += 1 / (60 + rank)
                raw_rows = sorted(fused.values(), key=lambda row: row["retrieval_score"], reverse=True)
        rows = []
        seen = set()
        for row in raw_rows:
            duplicate_key = (row["filename"], row["page"], row["text"])
            if duplicate_key in seen:
                continue
            seen.add(duplicate_key)
            rows.append(row)
            if len(rows) == k:
                break
        return rows
