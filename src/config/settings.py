from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Dataset settings
    limit: int | None = None  # None for all articles (~241,787)
    
    # Chunking settings
    chunk_size: int = 1000
    overlap: int = 200
    
    # Retrieval settings
    top_k: int = 5
    max_distance: float = 1.5
    context_chars_per_chunk: int = 1000
    
    # VectorDB settings
    collection_name: str = "wiki_simple"
    chroma_path: str = "chroma_db"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    
    # Generator settings: "gpt2" | "gemini" | "groq"
    generator_type: str = "gpt2"

settings = Settings()