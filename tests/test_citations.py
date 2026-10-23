from types import SimpleNamespace

import app.generation as generation
from app.generation import UNKNOWN_ANSWER, answer_question, _resolve_citations


def sources():
    return [
        {"source_id": "S1", "filename": "handbook.pdf", "page": 28,
         "text": "We offer one week of paid sick leave.", "distance": 0.2},
        {"source_id": "S2", "filename": "handbook.pdf", "page": 27,
         "text": "Employees receive 20 days of PTO per year.", "distance": 0.3},
    ]


def test_local_answer_has_real_source_citation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = answer_question("How much sick leave?", sources(), use_llm=False)
    assert "handbook.pdf, page 28" in result["answer"]
    assert [source["page"] for source in result["cited_sources"]] == [28]


def test_empty_evidence_abstains():
    result = answer_question("What is the lunch break?", [], use_llm=False)
    assert result["answer"] == UNKNOWN_ANSWER
    assert result["cited_sources"] == []


def test_local_fallback_returns_the_retrieved_evidence_passage():
    result = answer_question("How much sick leave?", sources(), use_llm=False)
    assert "one week of paid sick leave" in result["answer"].lower()
    assert result["cited_sources"][0]["page"] == 28


def test_frequency_question_prefers_the_policy_sentence_over_advance_notice():
    evidence = [{"source_id": "S1", "filename": "handbook.pdf", "page": 23,
                "distance": 0.2, "text": (
                    "Work from home. If your job doesn't require you to be present at our premises, "
                    "you can occasionally work from home (WFH). We normally allow [one day per week.] "
                    "Please inform your manager that you want to work from home using our HRIS at "
                    "least [two days] in advance. You can also set a recurring WFH day per week.")}]
    result = answer_question("How often may an eligible employee work from home?", evidence,
                             use_llm=False)
    assert "one day per week." in result["answer"]
    assert "[two days] in advance" not in result["answer"]
    assert result["cited_sources"][0]["page"] == 23


def test_local_fallback_does_not_infer_dental_coverage_from_health_insurance():
    related = [{"source_id": "S1", "filename": "handbook.pdf", "page": 22,
                "text": "At a minimum, we provide group health insurance to eligible employees.",
                "distance": 0.2}]
    result = answer_question("Does the handbook say dental insurance is included?", related, use_llm=False)
    assert result["answer"] == UNKNOWN_ANSWER
    assert result["cited_sources"] == []


def test_uncited_model_answer_is_not_returned(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Sick leave is 5 weeks."))],
                usage=SimpleNamespace(total_tokens=10))))

    monkeypatch.setattr(generation, "OpenAI", FakeClient)
    result = answer_question("How much sick leave?", sources())
    assert result["provider"] == "extractive fallback"
    assert "5 weeks" not in result["answer"]
    assert result["cited_sources"]


def test_model_citation_is_mapped_to_its_real_page(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="The allowance is one week [S1]."))],
                usage=SimpleNamespace(total_tokens=8))))

    monkeypatch.setattr(generation, "OpenAI", FakeClient)
    result = answer_question("How much sick leave?", sources())
    assert "handbook.pdf, page 28" in result["answer"]
    assert [source["page"] for source in result["cited_sources"]] == [28]


def test_invalid_model_citation_triggers_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Paid leave is 5 weeks [S9]."))],
                usage=SimpleNamespace(total_tokens=8))))

    monkeypatch.setattr(generation, "OpenAI", FakeClient)
    result = answer_question("How much sick leave?", sources())
    assert result["provider"] == "extractive fallback"
    assert "page 28" in result["answer"]
    assert "S9" not in result["answer"]


def test_only_retrieved_source_ids_resolve_to_real_document_pages():
    retrieved = [{"source_id": "S1", "filename": "policy.pdf", "page": 4, "text": "Evidence."}]
    rendered, cited = _resolve_citations("Supported [S1]; invented [S99].", retrieved)
    assert "policy.pdf, page 4" in rendered
    assert "S99" not in rendered
    assert [source["page"] for source in cited] == [4]


def test_compound_question_cites_both_retrieved_pages():
    evidence = [
        {"source_id": "S1", "filename": "handbook.pdf", "page": 27, "distance": 0.2,
         "text": "Employees receive 20 days of PTO each year."},
        {"source_id": "S2", "filename": "handbook.pdf", "page": 28, "distance": 0.25,
         "text": "We offer one week of paid sick leave."},
    ]
    result = answer_question("What annual PTO amount and paid sick leave benefit is described?",
                             evidence, use_llm=False)
    assert "20 days of PTO" in result["answer"]
    assert "paid sick leave" in result["answer"]
    assert {source["page"] for source in result["cited_sources"]} == {27, 28}


def test_exact_identifier_and_paraphrased_accident_query_get_cited_passages():
    evidence = [
        {"source_id": "S1", "filename": "handbook.pdf", "page": 23, "distance": 0.2,
         "text": "Please inform your manager to work from home using our HRIS at least two days in advance."},
        {"source_id": "S2", "filename": "handbook.pdf", "page": 25, "distance": 0.3,
         "text": "If you are involved in an accident in your company car, contact the HR department immediately."},
    ]
    hris = answer_question("Which exact system is named for submitting a work-from-home request?", evidence, use_llm=False)
    accident = answer_question("Find the phrase about company car accident reporting to HR.", evidence, use_llm=False)
    assert "HRIS" in hris["answer"]
    assert hris["cited_sources"][0]["page"] == 23
    assert "accident" in accident["answer"].lower()
    assert accident["cited_sources"][0]["page"] == 25
