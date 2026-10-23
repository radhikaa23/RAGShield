import pytest

from app.chunking import chunk_pages


class TinyTokenizer:
    """Deterministic tokenizer stub: one whitespace-separated item per token."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re

        offsets = [match.span() for match in re.finditer(r"\S+", text)]
        return {"offset_mapping": offsets}


def test_chunks_overlap_but_keep_source_page():
    pages = [{"page": 4, "text": "one two three four five six seven"}]
    chunks = chunk_pages(pages, size=4, overlap=1, tokenizer=TinyTokenizer())
    assert [chunk.text for chunk in chunks] == [
        "one two three four", "four five six seven"]
    assert [chunk.page for chunk in chunks] == [4, 4]
    assert [chunk.index for chunk in chunks] == [0, 1]


def test_token_chunks_preserve_original_case_and_punctuation():
    chunks = chunk_pages([{"page": 1, "text": "First Sentence. Second Sentence."}],
                         size=2, overlap=1, tokenizer=TinyTokenizer())
    assert chunks[0].text == "First Sentence."
    assert any("Second Sentence." in chunk.text for chunk in chunks)
    assert all(chunk.text[0].isupper() for chunk in chunks)


def test_chunks_do_not_merge_adjacent_pages():
    chunks = chunk_pages([
        {"page": 1, "text": "first page content"},
        {"page": 2, "text": "second page content"},
    ], size=20, overlap=2, tokenizer=TinyTokenizer())
    assert [chunk.page for chunk in chunks] == [1, 2]


@pytest.mark.parametrize("size,overlap", [(0, 0), (3, 3), (3, 4), (4, -1)])
def test_invalid_chunk_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], size=size, overlap=overlap, tokenizer=TinyTokenizer())


def test_default_configuration_is_500_tokens_with_75_token_overlap():
    from app.config import chunk_overlap, chunk_size

    assert chunk_size() == 500
    assert chunk_overlap() == 75


def test_token_chunks_observe_exact_window_and_overlap():
    words = [f"token{i}" for i in range(1100)]
    chunks = chunk_pages([{"page": 9, "text": " ".join(words)}],
                         size=500, overlap=75, tokenizer=TinyTokenizer())

    assert len(chunks) == 3
    assert [len(chunk.text.split()) for chunk in chunks] == [500, 500, 250]
    assert chunks[0].text.split()[-75:] == chunks[1].text.split()[:75]
    assert chunks[1].text.split()[-75:] == chunks[2].text.split()[:75]
    assert all(chunk.page == 9 for chunk in chunks)
