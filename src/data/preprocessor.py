import re
from functools import lru_cache
from typing import List


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

def simple_chunk(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Chunking function that splits text into fixed-size chunks with overlap.
    """
    text = (text or "").strip()
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    
    return chunks


@lru_cache(maxsize=2)
def _get_tokenizer(model_name: str = "gpt2"):
    try:
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    except ImportError:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    tokenizer.model_max_length = 1_000_000
    return tokenizer


def split_sentences(text: str) -> List[str]:
    """
    Split text into rough sentences.

    This is intentionally lightweight and regex-based so indexing stays simple.
    """
    text = clean_text(text)
    if not text:
        return []

    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    return sentences or [text]


def _split_long_sentence(
    sentence: str,
    max_tokens: int,
    token_overlap: int,
    tokenizer_name: str,
) -> List[str]:
    tokenizer = _get_tokenizer(tokenizer_name)
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        return [sentence]

    chunks = []
    start = 0
    overlap = min(token_overlap, max_tokens // 2)

    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
        if chunk:
            chunks.append(clean_text(chunk))
        if end == len(token_ids):
            break
        start = max(end - overlap, start + 1)

    return chunks


def hybrid_chunk(
    text: str,
    max_tokens: int = 180,
    overlap_sentences: int = 1,
    tokenizer_name: str = "gpt2",
    long_sentence_overlap_tokens: int = 20,
) -> List[str]:
    """
    Chunk text by sentences while enforcing a token budget.

    Strategy:
    1. clean text
    2. split into sentences
    3. group neighboring sentences until the token budget is reached
    4. keep a small sentence overlap between chunks

    If a single sentence is longer than the token budget, it is split by tokens.
    """
    text = clean_text(text)
    if not text:
        return []

    raw_sentences = split_sentences(text)
    expanded_sentences: List[str] = []
    sentence_token_counts: List[int] = []
    tokenizer = _get_tokenizer(tokenizer_name)

    for sentence in raw_sentences:
        token_count = len(tokenizer.encode(sentence, add_special_tokens=False))
        if token_count <= max_tokens:
            expanded_sentences.append(sentence)
            sentence_token_counts.append(token_count)
            continue

        long_parts = _split_long_sentence(
            sentence=sentence,
            max_tokens=max_tokens,
            token_overlap=long_sentence_overlap_tokens,
            tokenizer_name=tokenizer_name,
        )
        for part in long_parts:
            expanded_sentences.append(part)
            sentence_token_counts.append(len(tokenizer.encode(part, add_special_tokens=False)))

    if not expanded_sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_token_counts: List[int] = []
    current_tokens = 0

    for sentence, token_count in zip(expanded_sentences, sentence_token_counts):
        would_overflow = current_sentences and current_tokens + token_count > max_tokens

        if would_overflow:
            chunks.append(" ".join(current_sentences).strip())

            if overlap_sentences > 0:
                current_sentences = current_sentences[-overlap_sentences:]
                current_token_counts = current_token_counts[-overlap_sentences:]
                current_tokens = sum(current_token_counts)
            else:
                current_sentences = []
                current_token_counts = []
                current_tokens = 0

        current_sentences.append(sentence)
        current_token_counts.append(token_count)
        current_tokens += token_count

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return chunks


def clean_text(text: str) -> str:
    """
    Basic text cleaning function (can be extended later)
    """
    text = (text or "").strip()
    # Remove multiple spaces
    text = " ".join(text.split())
    return text