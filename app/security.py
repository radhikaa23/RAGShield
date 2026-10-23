import re

INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b", re.I),
    re.compile(r"\b(reveal|print|expose|leak|show)\b.{0,45}\b(api key|password|secret|token|system prompt)\b", re.I),
    re.compile(r"\b(disregard|override|bypass)\b.{0,45}\b(instructions|safety|policy|guardrails)\b", re.I),
    re.compile(r"\b(system prompt|developer message)\b.{0,60}\b(reveal|ignore|override|print)\b", re.I),
    re.compile(r"\b(read|retrieve|show|dump|disclose)\b.{0,50}\b(other|another|all)\b.{0,30}\b(documents?|users?|tenants?|files?)\b", re.I),
    re.compile(r"\b(repeat|output|show|reveal|print)\b.{0,50}\b(hidden (?:instructions?|prompts?)|environment variables?|credentials?)\b", re.I),
    re.compile(r"\b(treat|consider|follow)\b.{0,50}\b(document|text|content)\b.{0,40}\b(system|developer)\s+(?:instructions?|message)\b", re.I),
]
SECRET_PATTERNS = [
    re.compile(r"(?i)\b(?:password|api[_ -]?key|access[_ -]?token)\s*[:=]\s*\S{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
]


def inspect_text(text: str) -> dict:
    injection = any(p.search(text) for p in INJECTION_PATTERNS)
    secrets = [p.pattern for p in SECRET_PATTERNS if p.search(text)]
    return {"prompt_injection": injection, "secret_detected": bool(secrets),
            "secret_patterns": secrets}


def redact_secrets(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED SECRET]", text)
    return text


def validate_question(question: str) -> str:
    question = question.strip()
    if not question or len(question) > 1000:
        raise ValueError("Question must contain 1 to 1000 characters.")
    inspection = inspect_text(question)
    if inspection["secret_detected"]:
        raise ValueError("The question appears to contain a secret. Remove it before querying.")
    if inspection["prompt_injection"]:
        raise ValueError("This question contains a request to override instructions or access protected data.")
    return question


def safe_filename(filename: str) -> str:
    """Return a display-only basename; callers must still choose their own storage path."""
    from pathlib import PurePosixPath

    basename = PurePosixPath((filename or "").replace("\\", "/")).name
    basename = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", basename).strip(" .")
    if not basename or basename in {".", ".."}:
        return "uploaded.pdf"
    return basename[:180]
