from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Dataset settings
    limit: int | None = None  # None for all articles (~241,787)
    
    # Chunking settings
    chunk_size: int = 1000
    overlap: int = 200
    
    # Retrieval settings
    top_k: int = 3
    max_distance: float = 1.2
    context_chars_per_chunk: int = 300  # GPT-2 max input = 512 tokens; 3 chunks × 300 chars 
    small_to_big_enabled: bool = False
    small_to_big_window: int = 1
    lost_in_middle_mitigation: bool = True
    
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

    # GPT-2 generation parameters
    gpt2_model_name: str = "gpt2"
    gpt2_max_new_tokens: int = 50     # Short answers = faster inference
    # gpt2_do_sample: bool = True
    gpt2_do_sample: bool = False  # Disable sampling for more deterministic output
    gpt2_temperature: float = 0.2     # Near-deterministic = factual
    gpt2_top_p: float = 0.9
    gpt2_top_k: int | None = 50       # Limit vocabulary → more coherent text
    gpt2_no_repeat_ngram_size: int = 3  # Prevent repetitive output
    gpt2_max_input_length: int = 512  # GPT-2 hard context window limit
    
    # Prompt template for RAG
    # GPT-2 is a completion model, not instruction-following.
    # Shorter and simpler prompts work better — the model continues after "Answer:"
    prompt_template: str = """Context: {context}

Question: {query}
Answer:"""

settings = Settings()