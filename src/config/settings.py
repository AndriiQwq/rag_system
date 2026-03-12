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
    max_distance: float = 1.0
    context_chars_per_chunk: int = 1000
    
    # Additional retrieval settings
    use_reranking: bool = False  # Enable CrossEncoder reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3  # Number of documents after reranking
    
    # VectorDB settings
    collection_name: str = "wiki_simple"
    chroma_path: str = "chroma_db"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    
    # Generator settings: "gpt2" | "gemini" | "groq"
    generator_type: str = "gpt2"
    
    # Prompt template for RAG
    prompt_template: str = """Below is a context from Wikipedia and a question. Answer the question using ONLY the information from the context. Keep your answer short and factual.

    Context:
    {context}

    Question: {query}

    Answer:"""

settings = Settings()