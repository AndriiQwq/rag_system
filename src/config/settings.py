from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Dataset settings
    limit: int | None = None  # None for all articles (~241,787)
    
    # Chunking settings
    chunking_strategy: str = "hybrid"  # "hybrid" | "simple"
    chunk_size: int = 1000  # used by simple chunking
    overlap: int = 200  # used by simple chunking
    chunk_max_tokens: int = 180
    chunk_overlap_sentences: int = 1
    chunk_tokenizer_name: str = "gpt2"
    long_sentence_overlap_tokens: int = 20
    
    # Retrieval settings
    top_k: int = 3
    max_distance: float = 0.75
    context_chars_per_chunk: int = 500
    small_to_big_enabled: bool = False
    small_to_big_window: int = 1
    
    # Additional retrieval settings
    use_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 2  # Number of documents after reranking
    
    # VectorDB settings
    collection_name: str = "wiki_simple"
    chroma_path: str = "chroma_db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    
    # Generator settings: "gpt2" | "gemini" | "groq"
    generator_type: str = "gpt2"

    # GPT-2 generation parameters
    generation_model_name: str = "gpt2"
    generation_max_new_tokens: int = 40
    generation_do_sample: bool = False
    generation_temperature: float = 0.6
    generation_top_p: float = 0.8
    # generation_top_k: int | None = 40
    generation_no_repeat_ngram_size: int = None # can be None for no limit, or an int to prevent repeating n-grams of that size 
    generation_max_input_length: int = 512  
    
    # Prompt template for RAG
    # GPT-2 is a completion model, not instruction-following.
    # Shorter and simpler prompts work better — the model continues after "Answer:"
    prompt_template: str = """Context: {context}

Question: {query}
Answer:"""

    # External API
    groq_model_name: str = "llama-3.3-70b-versatile"
    gemini_model_name: str = "gemini-2.5-flash-lite"

settings = Settings()