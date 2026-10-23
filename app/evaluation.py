"""Run a reproducible K / chunk / retriever comparison on the bundled handbook."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from app.generation import UNKNOWN_ANSWER, answer_question
from app.ingestion import prepare_document
from app.retrieval import SecureIndex

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "evaluation_dataset.jsonl"
SAMPLE = ROOT / "data" / "sample_docs" / "Employee-Handbook.pdf"
CHUNK_CONFIGS = ((256, 38), (512, 75), (800, 120))
K_VALUES = (3, 5, 10)
STRATEGIES = ("semantic", "keyword", "hybrid")
DISTANCE_THRESHOLD = 0.72


def load_cases() -> list[dict]:
    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) < 30:
        raise ValueError("Evaluation dataset must contain at least 30 cases.")
    return cases


def _score(cases: list[dict], retrieved: dict[str, list[dict]], latency_samples: list[float],
           config: dict, use_llm: bool) -> dict:
    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]
    relevant_total = relevant_found = reciprocal_rank = page_hit_cases = 0
    answer_keyword_hits = answer_keyword_total = citation_hit = 0
    abstained = 0
    per_case = []
    for case in cases:
        rows = retrieved[case["id"]]
        expected_pages = set(case["relevant_pages"])
        ranks = [rank for rank, row in enumerate(rows, 1) if row["page"] in expected_pages]
        if case["answerable"]:
            relevant_total += len(expected_pages)
            relevant_found += len({row["page"] for row in rows if row["page"] in expected_pages})
            reciprocal_rank += 1 / ranks[0] if ranks else 0
            page_hit_cases += int(bool(ranks))
        for position, row in enumerate(rows, 1):
            row["source_id"] = f"S{position}"
        answer = answer_question(case["question"], rows, use_llm=use_llm)
        if case["answerable"]:
            keywords = case.get("answer_keywords", [])
            matched = [term for term in keywords if term.lower() in answer["answer"].lower()]
            answer_keyword_hits += len(matched)
            answer_keyword_total += len(keywords)
            citation_hit += int(any(source["page"] in expected_pages for source in answer["cited_sources"]))
        else:
            abstained += int(answer["answer"] == UNKNOWN_ANSWER)
        per_case.append({"id": case["id"], "answerable": case["answerable"],
            "expected_pages": sorted(expected_pages),
            "retrieved": [{"doc_id": row["doc_id"], "chunk_id": row["chunk_id"], "page": row["page"],
                           "distance": row["distance"], "score": row["retrieval_score"]} for row in rows],
            "answer": answer["answer"], "cited_pages": [s["page"] for s in answer["cited_sources"]]})
    return {
        **config,
        "retrieval_metrics": {
            "recall_at_k_page_references": round(relevant_found / max(relevant_total, 1), 3),
            "mrr_at_k": round(reciprocal_rank / max(len(answerable), 1), 3),
            "relevant_page_hit_rate": round(page_hit_cases / max(len(answerable), 1), 3),
            "mean_retrieval_latency_ms": round(1000 * statistics.mean(latency_samples), 2),
            "median_retrieval_latency_ms": round(1000 * statistics.median(latency_samples), 2),
        },
        "answer_metrics": {
            "answer_keyword_coverage_heuristic": round(answer_keyword_hits / max(answer_keyword_total, 1), 3),
            "citation_page_hit_rate": round(citation_hit / max(len(answerable), 1), 3),
            "unsupported_abstention_rate": round(abstained / max(len(unanswerable), 1), 3),
            "answer_generator": "configured LLM" if use_llm else "local extractive fallback",
            "method_note": "Keyword coverage and expected-page checks are proxies; they do not establish faithfulness. Inspect per-case answers and passages.",
        },
        "cases": per_case,
    }


def run_matrix(use_llm: bool = False) -> dict:
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in .env before running with --with-llm.")
    cases = load_cases()
    comparisons = []
    # Independent in-memory Chroma stores prevent experiments from changing the app's index.
    for size, overlap in CHUNK_CONFIGS:
        prepared = prepare_document(SAMPLE, "evaluation-owner", size=size, overlap=overlap)
        index = SecureIndex(directory=":memory:")
        index.upsert(prepared["chunks"])
        for strategy in STRATEGIES:
            for k in K_VALUES:
                retrieved = {}
                latency_samples = []
                for case in cases:
                    start = time.perf_counter()
                    retrieved[case["id"]] = index.search(case["question"], "evaluation-owner",
                        [prepared["doc_id"]], k=k, distance_threshold=DISTANCE_THRESHOLD,
                        strategy=strategy)
                    latency_samples.append(time.perf_counter() - start)
                comparisons.append(_score(cases, retrieved, latency_samples,
                    {"chunk_size_tokens": size, "chunk_overlap_tokens": overlap, "k": k,
                     "retrieval_strategy": strategy, "cosine_distance_threshold": DISTANCE_THRESHOLD}, use_llm))
    return {"dataset_cases": len(cases), "answerable_cases": sum(c["answerable"] for c in cases),
        "unsupported_cases": sum(not c["answerable"] for c in cases),
        "chunk_configurations": [{"size_tokens": s, "overlap_tokens": o} for s, o in CHUNK_CONFIGS],
        "k_values": list(K_VALUES), "strategies": list(STRATEGIES),
        "metric_definitions": {"Recall@K": "fraction of manually labeled relevant PDF page references present in top K (a page proxy because chunk IDs vary by chunk configuration)",
            "MRR": "mean reciprocal rank of the first retrieved chunk on an expected page", "latency": "per-query retrieval wall time; mean and median are reported; includes semantic query embedding, excludes index construction and answer generation"},
        "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm", action="store_true", help="Also evaluate answers with configured hosted LLM; provider charges may apply.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path (default prints to terminal).")
    args = parser.parse_args()
    result = run_matrix(use_llm=args.with_llm)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote evaluation report: {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
