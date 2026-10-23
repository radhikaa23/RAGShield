# RAGShield evaluation baseline

Historical measured baseline from before the retrieval comparison matrix was added. It used the bundled `Employee-Handbook.pdf`, 32 cases, 500-token chunks, 75-token overlap, K=5 semantic retrieval, and the local extractive answer path. The current dataset has 35 cases. Reproduce the current matrix with `python -m app.evaluation --output docs/evaluation-report.json`; do not treat these older values as measurements of the current implementation.

## Default distance cutoff: 0.72

| Measure | Result |
| --- | ---: |
| Answerable cases with a relevant page in top 5 (Hit Rate@5) | 0.852 (23/27) |
| MRR@5 | 0.765 |
| Unanswerable cases that still retrieved a passage | 1.000 (5/5) |
| False answers on those unanswerable cases, local path | 0.000 (0/5) |
| Correct-page citations among answerable cases | 0.667 (18/27) |
| Answer-keyword coverage heuristic | 0.410 |
| Mean retrieval latency | 93.59 ms/query |

All five unanswerable cases were answered with the configured abstention text, even though retrieval returned semantically related passages. This shows why a distance threshold alone is not a reliable test that evidence answers a question. The threshold sweep found no cutoff satisfying both the project's answerable hit-rate and unanswerable retrieval criteria. The `0.72` default is therefore an explicit starting point, not a calibrated optimum.

Keyword coverage and page-hit rate are evaluation proxies. They do not prove that a response is entailed by its cited text. Review individual retrieved passages and answers before using this project with real policies. The included handbook is a generic template with unresolved bracketed values.
