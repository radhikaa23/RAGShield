# RAGShield security test report

**Run date:** 2026-10-23  
**Command:** `python -m pytest -q`  
**Result:** **71 passed, 1 skipped, 0 failed**  
**Warning:** One upstream ChromaDB deprecation warning (`asyncio.iscoroutinefunction`).

## Covered by the deterministic run

- Known prompt-injection attempts in questions and retrieved chunks are detected/rejected or excluded before generation.
- Fake-provider checks verify document evidence is placed in the user evidence message, not the system message, and a filtered injection chunk does not enter model context.
- Secret-like strings are redacted; tests assert their original values are absent from generated answers.
- PDF signature and parser validation, invalid extension despite a PDF MIME claim, oversized uploads, malformed PDFs, encrypted PDFs, textless PDFs, page limits, extracted-text limits, and path traversal display names.
- Startup rejects a missing/example/short access token and invalid upload-size configuration.
- ID tampering is rejected before retrieval; request-supplied `owner_id` does not change the authenticated identity; unauthorized deletion is denied.
- Semantic, keyword, and hybrid result sets are post-filtered for both owner and selected document ID, even when a test collection deliberately returns unfiltered rows.
- Authorized retrieval returns a cited answer; deletion filters by both owner and document ID.
- Legacy upload cleanup removes only a matching prior-build file after verifying its owner-scoped content ID.

Tests call route handlers directly with fake indexes and use fake model clients for boundary tests; they do not contact a model provider or run an ASGI server. Existing PDF extraction, chunking, retrieval, and citation tests also ran in the same suite.

## Skipped live-model check

`tests/test_security_llm.py` was skipped because live security tests are opt-in. Run it only when ready to send a provider request:

```powershell
$env:RAGSHIELD_RUN_LLM_SECURITY_TESTS = "1"
python -m pytest -m llm_security
```

Set `OPENAI_API_KEY` in the environment first. This test can incur a model charge and its result can vary. A pass applies only to that particular provider/model/run and does not establish complete prompt-injection protection.

## Interpretation

The deterministic suite verifies implemented controls and expected failure behavior; it is not a penetration test or security certification. Pattern-based detection can miss novel/obfuscated instructions. The single-token identity is local-development-only. See [SECURITY.md](../SECURITY.md) for the threat model, mitigated/partial/out-of-scope items, and reporting instructions.
