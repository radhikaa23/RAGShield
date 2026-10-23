# RAGShield — Secure Document Intelligence

RAGShield is a local PDF question-answering portfolio project. It extracts text from the supplied employee handbook, retrieves relevant passages, and answers with the PDF filename and page number. It also demonstrates the security controls requested for the original RAGShield project.

> **About the sample handbook:** This is a generic template, not a finalized company policy. It contains bracketed options such as `[one week]` and `[can/cannot]`. RAGShield preserves those placeholders; do not treat them as confirmed policy.

## What is in the project

- **Ingestion:** PyMuPDF validates and extracts page-numbered text. Model-token chunks preserve page metadata. Content-based IDs make repeated uploads idempotent.
- **Embeddings and storage:** Sentence Transformers `all-MiniLM-L6-v2` creates local embeddings. ChromaDB stores chunks persistently in `data/chroma/`.
- **Retrieval:** Chroma performs cosine-distance semantic search. A dependency-free BM25 scorer handles exact words and identifiers; hybrid mode combines both ranked lists using Reciprocal Rank Fusion (RRF). The UI/API accept K=3, 5, or 10 and apply the configured cosine-distance cutoff to semantic candidates.
- **Answers and citations:** An optional OpenAI-compatible chat model answers from retrieved evidence. Without an API key, RAGShield quotes a relevant passage rather than making a model-generated claim. The model must cite evidence IDs; IDs are mapped to real filename/page metadata. Retrieved passages and passages cited by the answer are displayed separately.
- **Interface:** Streamlit handles upload and chat; FastAPI provides upload, query, owner-scoped document listing/deletion, and bearer-token checks. The token maps to the fixed `local-owner` development identity; an `owner_id` in a request is ignored and never establishes identity.
- **Evaluation:** The 35-case handbook dataset includes supported and unsupported questions, exact identifiers, and a two-page multi-fact case. The matrix evaluates three token chunk settings, K=3/5/10, and semantic/BM25/hybrid retrieval on the same cases. It writes retrieval metrics separately from answer and citation proxies.
- **Security controls:** Retrieved PDF text is treated as untrusted data. Pattern-matched injection chunks are filtered at retrieval and again before generation; likely secrets are redacted; owner/document filters are checked by Chroma and again in application code. The single bearer token is a local-development identity mode, not production authentication.

## Project layout

```text
app/
  chunking.py       page-preserving chunk creation
  config.py         environment-backed RAG settings
  evaluation.py     chunk/K/retriever experiment matrix
  generation.py     LLM and extractive answer paths
  ingestion.py      PDF validation, extraction, IDs, and chunk metadata
  main.py           FastAPI upload, document, and query routes
  retrieval.py      persistent ChromaDB vector search
  security.py       input checks, injection detection, secret redaction
frontend/
  streamlit_app.py  upload and chat interface
tests/              deterministic security, retrieval, upload, chunking, and citation tests
  test_security_llm.py opt-in provider-backed prompt-injection check
data/
  sample_docs/Employee-Handbook.pdf
  evaluation_dataset.jsonl 35 labeled question cases
docs/architecture.md
docs/security-test-report.md
SECURITY.md
```

## Run on Windows

Python 3.11 or newer is recommended. Run these commands from the project folder in PowerShell. If you already created `.venv`, activate it and continue at the install step.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` once from the example, then replace the access token with a private random value of at least 32 characters. Keep `.env` private; it is ignored by Git. The sample LLM key is blank, so no paid API is called by default. For deployment, use a secret manager and inject variables at runtime rather than copying `.env` to a server.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

To create a random local bearer token in PowerShell, run this and copy its output into the `RAGSHIELD_ACCESS_TOKEN` value in `.env`:

```powershell
$tokenBytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($tokenBytes)
[Convert]::ToBase64String($tokenBytes)
```

Run the API in the first activated PowerShell window:

```powershell
python -m uvicorn app.main:app --reload
```

Run Streamlit in a second activated PowerShell window:

```powershell
python -m streamlit run frontend/streamlit_app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`) and enter the same token from `.env` in the sidebar. The sample handbook is indexed automatically at API startup. On first use, Sentence Transformers downloads `all-MiniLM-L6-v2`; this needs internet access and may take a few minutes. ChromaDB persists embeddings locally, so subsequent starts reuse the index.

### Optional LLM setup and cost

Set `OPENAI_API_KEY` in `.env` and choose `OPENAI_MODEL` to use an OpenAI-compatible API. Hosted model calls can incur token-based charges; check your provider's current pricing before enabling them. The local Sentence Transformers embedder has no per-request embedding API charge. With no API key, the built-in extractive answer path works locally. A local OpenAI-compatible server can also be used by setting `OPENAI_BASE_URL` and `OPENAI_MODEL` to that server's endpoint and installed model.

## Configuration

| Setting | Default | Purpose |
| --- | ---: | --- |
| `RAGSHIELD_CHUNK_SIZE` | `500` | Maximum model-token window per chunk; chunking is token-based, not character-based |
| `RAGSHIELD_CHUNK_OVERLAP` | `75` | Shared model tokens between adjacent chunks on the same page |
| `RAGSHIELD_TOP_K` | `5` | Maximum retrieved chunks per question |
| `RAGSHIELD_MAX_DISTANCE` | `0.72` | Maximum Chroma cosine distance; lower is stricter |
| `RAGSHIELD_MAX_PDF_MB` | `20` | Maximum upload size, 1–100 MB |
| `RAGSHIELD_MAX_PDF_PAGES` | `300` | Maximum PDF page count, 1–1000 |
| `RAGSHIELD_MAX_EXTRACTED_CHARS` | `5000000` | Maximum extracted text characters per PDF |
| `OPENAI_MODEL` | `gpt-4o-mini` | Optional answer-generation model |
| `RAGSHIELD_ACCESS_TOKEN` | required | Local API bearer token; startup rejects the example placeholder and values shorter than 32 characters |

For cosine distance, `0` is identical and larger values are less similar. The default cutoff is an initial value. Use the evaluation sweep below to select a cutoff based on this dataset and then set it in `.env`; results should be recalculated if the corpus or embedding model changes.

## Evaluation and tests

After installing requirements and from the project root:

```powershell
python -m app.evaluation --output docs/evaluation-report.json
python -m pytest
```

The evaluation builds isolated in-memory Chroma indexes, so it does not rewrite `data/chroma`. It evaluates 27 combinations (3 chunk settings × 3 K values × 3 retrieval methods) over the same 35 questions. Recall@K is page-reference recall because chunk IDs change between chunk settings; MRR ranks the first retrieved chunk on a labeled page. Retrieval latency reports mean and median per query; it includes semantic query embedding but excludes index construction and answer generation. Answer keyword coverage, citation page hits, and abstention rates are separate heuristics; they do not prove answer faithfulness. The full per-case JSON helps inspect errors. Add `--with-llm` to evaluate the configured generator; provider charges may apply.

The regular suite is deterministic and makes no provider calls. To opt into the separate live-model security probe, set `RAGSHIELD_RUN_LLM_SECURITY_TESTS=1` in the shell and provide `OPENAI_API_KEY`, then run `python -m pytest -m llm_security`. Live behavior can vary and provider charges may apply. The latest deterministic results and skipped live checks are in [docs/security-test-report.md](docs/security-test-report.md).

## Deploy a read-only portfolio demo

The repository includes `render.yaml` for the FastAPI backend. Public demo mode disables upload and deletion at both the UI and API layers. A fresh backend indexes the bundled sample handbook at startup. This mode is still a shared demo identity; do not upload or index private documents, and do not present it as multi-user authentication.

1. Confirm `.env` is not tracked (`git check-ignore -v .env`) and is absent from the GitHub file list. Keep `.env` and all real tokens private. If a token was ever pushed or shown publicly, rotate it.
2. Push the deployment files to GitHub: `app/main.py`, `frontend/streamlit_app.py`, `render.yaml`, and `README.md`.
3. In Render, choose **New + → Blueprint**, connect this repository, and create the service from `render.yaml`. When prompted, set `RAGSHIELD_ACCESS_TOKEN` to a fresh random value of at least 32 characters. Do not put the token in the YAML or GitHub. Wait for deploy and confirm `https://<your-render-service>.onrender.com/health` returns `{"status":"ok",...}`.
4. In Streamlit Community Cloud, create an app from this repository with entrypoint `frontend/streamlit_app.py`.
5. In that Streamlit app's **Settings → Secrets**, add the following TOML, replacing the URL with your Render service URL and the token with the same secret you configured in Render:

   ```toml
   RAGSHIELD_API_URL = "https://<your-render-service>.onrender.com"
   RAGSHIELD_ACCESS_TOKEN = "<same-private-token-as-Render>"
   RAGSHIELD_DEMO_MODE = "true"
   ```

6. Save/reboot the Streamlit app. Its `streamlit.app` URL is the shareable UI link. Check that the sample handbook answers questions and that the UI has no upload control. The API also rejects upload and delete requests while demo mode is enabled.

The backend uses local Chroma storage. On a host with an ephemeral filesystem, uploaded/indexed data can be lost on restart or redeploy; this read-only demo rebuilds its sample index at startup. For production or private documents, configure persistent storage and real per-user authentication before deployment.

### Measured comparison (local run, 35 cases)

With 512/75 chunks and K=5, semantic retrieval gave Recall@K **1.000**, MRR **0.942**, and median retrieval latency **34.95 ms**; BM25 gave **0.903**, **0.823**, and **39.10 ms**; RRF hybrid gave **0.968**, **0.933**, and **73.35 ms**. The local answer fallback's citation page-hit rates were 0.867, 0.700, and 0.833 respectively. Unsupported-question abstention was 1.000 for semantic, 0.600 for BM25, and 0.800 for hybrid. This shows the practical risk of lexical matching without a calibrated relevance threshold; treat these as one local run, not a general performance guarantee.

| Semantic settings | Recall@K | MRR | Median retrieval latency |
| --- | ---: | ---: | ---: |
| 256 tokens / 38 overlap, K=5 | 0.968 | 0.883 | 39.39 ms |
| 512 tokens / 75 overlap, K=5 | 1.000 | 0.942 | 34.95 ms |
| 800 tokens / 120 overlap, K=5 | 1.000 | 0.942 | 36.44 ms |

| Semantic 512/75 | Recall@K | MRR | Median retrieval latency |
| --- | ---: | ---: | ---: |
| K=3 | 0.968 | 0.933 | 39.65 ms |
| K=5 | 1.000 | 0.942 | 34.95 ms |
| K=10 | 1.000 | 0.942 | 43.54 ms |

On this dataset K=5 recovered all labeled pages; K=10 did not improve recall and was slower at the median. Larger K usually exposes more evidence and can increase answer-context size and latency, but gains depend on the corpus. The 800-token row is limited by the current embedding model's 512-token input cap as described below.

Chunking uses the tokenizer for `all-MiniLM-L6-v2`, not character counts. That model has a 512-token input limit, so the 800-token experiment preserves 800-token text chunks but the model may truncate input during embedding. Treat that run as diagnostic; for a fair 800-token embedding comparison, switch to a longer-context embedding model and rebuild the index. Larger K can improve recall by exposing more candidate passages, while increasing prompt size, generation latency, and the chance of noisy evidence. Smaller chunks can improve pinpoint retrieval but may split facts; overlap helps preserve context at the cost of redundant storage and candidates.

## Demo queries

Try questions such as:

- “How much sick leave does the handbook offer?” — expected evidence: PDF page 28, with the template's `[one week]` placeholder.
- “How many days of PTO do employees receive per year?” — expected evidence: PDF page 27, including `[20 days]`.
- “How often may an eligible employee work from home?” — expected evidence: PDF page 23, including `[one day per week]`.
- “How long is the lunch break?” — the sample does not specify this; the expected behavior is an abstention when retrieval finds no sufficiently relevant passage.

The exact wording of extractive fallback answers depends on the text extraction and retrieved chunk, so citations and passage display are the auditable output.

## Known limitations and next security work

- The sample is a template with unresolved bracketed options and is not company-specific policy.
- Scanned PDFs need OCR, which is not included. Upload size, page count, and extracted-text limits can be configured in `.env`.
- Injection and secret checks are pattern-based; they can miss novel content or flag benign text. Prompting is not a guarantee against injection.
- The local bearer token maps every authorized request to one demo owner. It is local-development authentication, not multi-user identity management or production access control. Do not expose this app publicly until real user/tenant identity is integrated.
- The API sends selected evidence to the configured model provider when an LLM key is set. Review provider privacy and retention settings before using confidential files.
- Answer keyword coverage and citation-page hits are evaluation proxies. A human should verify that each cited passage actually supports its claim.

Useful Level 2 follow-up work: stronger adversarial document tests, authenticated multi-user identities, per-user key management, richer secret scanning, citation entailment review, and red-team evaluation.

## Interview discussion prompts

1. Why does retrieval improve document QA compared with putting an entire PDF in a prompt?
2. How do chunk size and overlap affect retrieval precision and context continuity?
3. What does Chroma's cosine distance mean, and how was the cutoff selected?
4. How do content-based document IDs make repeated uploads idempotent?
5. How do source IDs prevent the model from inventing filename/page citations?
6. Why is a source citation not proof that an answer is faithful to the passage?
7. What would you change to support multiple users safely?
8. What are the limits of pattern-based prompt-injection detection?

See [docs/architecture.md](docs/architecture.md) for the data flow.
