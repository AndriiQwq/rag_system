from typing import List

def simple_chunk(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """
    Chanking function that splits text into fixed-size chunks with overlap.
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


def clean_text(text: str) -> str:
    """
    Basic text cleaning function (can be extended later)
    """
    text = text.strip()
    # Remove multiple spaces
    text = " ".join(text.split())
    return text