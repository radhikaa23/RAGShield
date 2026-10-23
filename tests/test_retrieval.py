import pytest

from app.retrieval import SecureIndex, _bm25


class FakeCollection:
    def __init__(self):
        self.rows = [
            {"text": "irrelevant injection", "meta": {"page": 1, "filename": "a.pdf", "doc_id": "a", "chunk_id": "1", "owner_id": "owner-a", "injection_flag": True}, "distance": 0.1},
            {"text": "closest passage", "meta": {"page": 2, "filename": "a.pdf", "doc_id": "a", "chunk_id": "2", "owner_id": "owner-a", "injection_flag": False}, "distance": 0.2},
            {"text": "weak passage", "meta": {"page": 3, "filename": "a.pdf", "doc_id": "a", "chunk_id": "3", "owner_id": "owner-a", "injection_flag": False}, "distance": 0.9},
        ]

    def count(self):
        return len(self.rows)

    def query(self, **kwargs):
        return {
            "documents": [[row["text"] for row in self.rows]],
            "metadatas": [[row["meta"] for row in self.rows]],
            "distances": [[row["distance"] for row in self.rows]],
        }

    def get(self, **kwargs):
        if "limit" in kwargs:
            return {"metadatas": [{"filename": "a.pdf", "doc_id": "a", "owner_id": "owner-a"}]}
        return {"metadatas": []}


def test_search_filters_injection_and_weak_matches_and_returns_metadata():
    index = SecureIndex.__new__(SecureIndex)
    index.collection = FakeCollection()
    rows = index.search("question", "owner-a", ["a"], k=2, distance_threshold=0.72)
    assert len(rows) == 1
    assert rows[0]["text"] == "closest passage"
    assert rows[0]["page"] == 2
    assert rows[0]["chunk_id"] == "2"
    assert rows[0]["doc_id"] == "a"
    assert rows[0]["distance"] == 0.2
    assert rows[0]["cosine_similarity"] == 0.8


@pytest.mark.parametrize("k", [3, 5, 10])
def test_search_accepts_requested_top_k(k):
    index = SecureIndex.__new__(SecureIndex)
    index.collection = FakeCollection()
    assert len(index.search("question", "owner-a", ["a"], k=k, distance_threshold=0.72)) == 1


def test_cosine_distance_threshold_abstains_from_weak_matches():
    index = SecureIndex.__new__(SecureIndex)
    index.collection = FakeCollection()
    assert index.search("question", "owner-a", ["a"], k=3, distance_threshold=0.1) == []


def test_bm25_prefers_exact_identifier_and_error_terms():
    scores = _bm25("HRIS ERR-492", ["Request access through HRIS after ERR-492", "General remote work guidance"])
    assert scores[0] > scores[1] == 0


def test_document_exists_check_is_owner_scoped():
    index = SecureIndex.__new__(SecureIndex)
    index.collection = FakeCollection()
    assert index.has_document("owner-a", "a")
    assert not index.has_document("owner-b", "a")


def test_document_deletion_filter_is_scoped_to_owner_and_document_id():
    class DeleteCollection:
        def delete(self, **kwargs):
            self.kwargs = kwargs

    index = SecureIndex.__new__(SecureIndex)
    index.collection = DeleteCollection()
    index.delete_document("owner-a", "a")
    assert index.collection.kwargs["where"] == {
        "$and": [{"owner_id": "owner-a"}, {"doc_id": "a"}]}


def test_document_list_hides_stale_legacy_copy_when_current_document_exists():
    class DuplicateDocuments:
        def get(self, **kwargs):
            return {"metadatas": [
                {"doc_id": "legacy-id", "filename": "Employee-Handbook.pdf", "owner_id": "owner-a"},
                {"doc_id": "new-id", "filename": "Employee-Handbook.pdf",
                 "content_hash": "same-bytes", "chunking_version": 2, "owner_id": "owner-a"},
            ]}

    index = SecureIndex.__new__(SecureIndex)
    index.collection = DuplicateDocuments()
    assert index.documents("owner-a") == {"new-id": "Employee-Handbook.pdf"}


@pytest.mark.parametrize("strategy", ["semantic", "keyword", "hybrid"])
def test_every_search_strategy_postfilters_owner_and_selected_document(strategy):
    class UnfilteredCollection:
        rows = [
            {"text": "needle authorized policy", "meta": {"page": 1, "filename": "mine.pdf", "doc_id": "mine", "chunk_id": "mine:1", "owner_id": "owner-a", "injection_flag": False}, "distance": 0.1},
            {"text": "needle unauthorized secret", "meta": {"page": 9, "filename": "other.pdf", "doc_id": "other", "chunk_id": "other:1", "owner_id": "owner-b", "injection_flag": False}, "distance": 0.1},
            {"text": "needle unselected content", "meta": {"page": 8, "filename": "hidden.pdf", "doc_id": "unselected", "chunk_id": "unselected:1", "owner_id": "owner-a", "injection_flag": False}, "distance": 0.1},
        ]

        def count(self):
            return len(self.rows)

        def query(self, **kwargs):
            return {"documents": [[r["text"] for r in self.rows]],
                    "metadatas": [[r["meta"] for r in self.rows]],
                    "distances": [[r["distance"] for r in self.rows]]}

        def get(self, **kwargs):
            return {"documents": [r["text"] for r in self.rows],
                    "metadatas": [r["meta"] for r in self.rows]}

    index = SecureIndex.__new__(SecureIndex)
    index.collection = UnfilteredCollection()
    rows = index.search("needle", "owner-a", ["mine"], k=10, strategy=strategy)
    assert rows
    assert all(row["doc_id"] == "mine" for row in rows)
    assert all("unauthorized" not in row["text"] for row in rows)
