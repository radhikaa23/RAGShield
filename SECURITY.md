# RAGShield security notes

## Scope and threat model

RAGShield accepts user-supplied PDFs and questions, stores extracted text and metadata in local ChromaDB, and can send selected passages to an optional OpenAI-compatible model. The protected assets are document contents, document ownership metadata, the local bearer token, and optional provider credentials. The main trust boundaries are the upload/API boundary, Chroma retrieval, and the LLM prompt boundary.

The current identity mechanism is deliberately a **local-development mode**: one configured bearer token maps every authorized request to the fixed `local-owner`. It demonstrates server-side authorization checks for a portfolio project. It is not multi-user authentication and must not be exposed as a public service.

## Controls and status

| Threat/control | Status | Enforcement and limits |
| --- | --- | --- |
| Untrusted PDF instructions | Partially mitigated | Pattern-matched chunks are excluded during retrieval and again before generation. The system message separates policy from document evidence. Novel or obfuscated instructions may evade patterns, and model prompting is not a proof of safety. |
| User attempts to override policy or reveal protected prompts/secrets | Partially mitigated | Known patterns are rejected before retrieval. Detection is lexical and may have false positives/negatives. |
| Cross-document or cross-owner retrieval | Mitigated in this local single-owner mode | The server derives owner identity from the bearer token, checks selected IDs against owner-scoped document metadata, passes owner/document filters to Chroma for semantic, keyword, and hybrid retrieval, and post-filters returned metadata. Authorization is never delegated to the LLM. |
| Document ID tampering | Mitigated for current API routes | Query and deletion first verify the document under the authenticated owner. Unknown and other-owner IDs return the same not-found/forbidden style response. |
| Unauthorized generation context | Mitigated by pipeline controls | Only retrieval results that pass owner/document and injection filters are passed to generation. Generation applies a second injection filter and secret redaction. No answer/retrieval cache is implemented. |
| Unauthorized deletion | Mitigated for indexed data | Deletion verifies owner-scoped metadata and Chroma deletion uses both owner and document ID. New upload bytes are temporary and deleted after indexing. On re-index or delete, a narrowly matched legacy upload file is removed only after its content ID is verified for that owner. |
| Malformed, oversized, encrypted, or textless PDFs | Partially mitigated | Signature and parser checks, configurable size/page/extracted-character limits, password rejection, and safe errors are applied. OCR is not provided. |
| Path traversal and active PDF content | Partially mitigated | Display filenames are reduced to basenames; disk upload names are random server-generated names. PyMuPDF extracts text only; this application does not execute PDF scripts or extract/run attachments. No malware scanner is included. |
| Secret leakage | Partially mitigated | Common key/password patterns are redacted at ingestion and again before generation; question validation rejects detected secret values; error logs include exception types, not exception messages. Pattern scanners cannot recognize every secret format. |
| API key/token storage | Mitigated for local setup only | Credentials are read from environment variables; `.env` is ignored by Git and `.env.example` contains placeholders. Use a deployment secret manager in production. |

## Upload behavior

The upload handler ignores browser MIME type as proof of file type. It requires a `.pdf` name and `%PDF-` signature, parses the bytes with PyMuPDF, rejects encrypted or textless PDFs, and enforces `RAGSHIELD_MAX_PDF_MB`, `RAGSHIELD_MAX_PDF_PAGES`, and `RAGSHIELD_MAX_EXTRACTED_CHARS`. The original uploaded bytes are written to a random temporary filename only while indexing and removed afterward. Older builds could leave a file in `data/uploads`; re-indexing or deleting its matching document removes only a legacy-pattern file whose content ID matches the authenticated owner and document. Chroma retains extracted text and metadata until the document is deleted.

Text extraction is not visual inspection. White-on-white, tiny-font, off-page, image-only, and other layout-based tricks are not comprehensively classified. Image-only PDFs are rejected as textless; OCR and visual risk analysis are out of scope.

## Secret handling

- Never commit `.env`, provider API keys, or real bearer tokens.
- At startup, the API requires a non-placeholder `RAGSHIELD_ACCESS_TOKEN` of at least 32 characters.
- Keep `.env` for local development only. For deployments, inject secrets from the platform's secret manager and rotate them when exposed.
- The optional model provider receives only the evidence returned for the authorized query. Review provider privacy, retention, and residency before using confidential data.
- Logs and HTTP errors avoid raw exception text. Do not add request bodies, tokens, or document text to logs.

## Tests

The default automated tests are deterministic and use fake model responses for citation/prompt-boundary checks. They cover malicious questions and chunks, redaction, upload validation and limits, owner/document filtering in each retrieval mode, ID tampering, authorization on deletion, and successful authorized retrieval.

`tests/test_security_llm.py` is an opt-in provider-backed probe. Set `RAGSHIELD_RUN_LLM_SECURITY_TESTS=1` and `OPENAI_API_KEY` before running `python -m pytest -m llm_security`. It makes a live model request, may incur charges, and can vary across providers/model versions. A pass is evidence for that run only, not a guarantee.

## Out of scope

Production identity and tenant management, TLS termination, rate limiting, brute-force protection, encrypted-at-rest Chroma storage, backup/retention controls, malware scanning, OCR/visual PDF analysis, advanced secret detection, robust defenses against novel prompt injection, and formal security certification are not implemented. The bearer-token demo should be placed behind a trusted local boundary only.

## Reporting a vulnerability

For a private project, report suspected issues directly to the project maintainer through a private channel. Include the affected endpoint/file, a minimal reproduction, and expected versus actual behavior. Do not include real API keys or other people's documents; replace them with synthetic data. Allow the maintainer time to investigate and release a fix before public disclosure. This portfolio project has no formal bug-bounty or guaranteed response-time program.
