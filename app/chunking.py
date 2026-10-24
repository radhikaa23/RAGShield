from dataclasses import dataclass
from functools import lru_cache


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MAX_TOKENS = 512
# Increment when chunk or embedding behavior changes so seeded indexes rebuild.
CHUNKING_VERSION = 3


@lru_cache(maxsize=1)
def _load_tokenizer():
    """Use the same tokenizer as the embedding model for real token windows."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL, use_fast=True)


@dataclass
class Chunk:
    text: str
    page: int
    index: int


def chunk_pages(pages: list[dict], size: int = 500, overlap: int = 75,
                tokenizer=None) -> list[Chunk]:
    """Split each page into model-token windows while preserving page citations."""
    if size < 1 or overlap < 0 or size <= overlap:
        raise ValueError("size must be positive and greater than non-negative overlap")
    tokenizer = tokenizer or _load_tokenizer()
    out, index = [], 0
    for item in pages:
        source_text = item["text"]
        encoded = tokenizer(source_text, add_special_tokens=False,
                            return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        start = 0
        while start < len(offsets):
            end = min(start + size, len(offsets))
            if end < len(offsets):
                # Prefer a nearby sentence boundary, while never exceeding size.
                lower_bound = max(start + 1, end - overlap)
                for candidate in range(end, lower_bound - 1, -1):
                    boundary = offsets[candidate - 1][1]
                    tail = source_text[boundary:boundary + 4].lstrip()
                    last_char = source_text[:boundary].rstrip()[-1:]
                    if tail and last_char in ".!?":
                        end = candidate
                        break
            start_char = offsets[start][0]
            end_char = offsets[end - 1][1]
            text = source_text[start_char:end_char].strip()
            if text:
                out.append(Chunk(text=text, page=item["page"], index=index))
                index += 1
            if end >= len(offsets):
                break
            start = max(start + 1, end - overlap)
    return out
