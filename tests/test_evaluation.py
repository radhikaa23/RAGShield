import pytest

from app.evaluation import CHUNK_CONFIGS, K_VALUES, STRATEGIES, load_cases, run_matrix


def test_evaluation_dataset_covers_supported_unsupported_and_multi_page_cases():
    cases = load_cases()
    assert len(cases) >= 30
    assert any(case["answerable"] for case in cases)
    assert any(not case["answerable"] for case in cases)
    assert any(len(case["relevant_pages"]) > 1 for case in cases if case["answerable"])
    assert any(case.get("case_type") == "exact_identifier" for case in cases)
    assert all("relevant_pages" in case for case in cases)


def test_matrix_has_requested_chunk_sizes_k_values_and_retrievers():
    assert [size for size, _ in CHUNK_CONFIGS] == [256, 512, 800]
    assert K_VALUES == (3, 5, 10)
    assert STRATEGIES == ("semantic", "keyword", "hybrid")


def test_llm_evaluation_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Set OPENAI_API_KEY"):
        run_matrix(use_llm=True)
