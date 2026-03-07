from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    limit: int | None = 500
    chunk_size: int = 1000
    overlap: int = 200
    collection_name: str = "wiki_simple"
    chroma_path: str = "chroma_db"
    
    # Generator settings: "gpt2" | "gemini" | "groq"
    generator_type: str = "gpt2"

settings = Settings()