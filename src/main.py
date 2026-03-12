import argparse
import time

from .config.settings import settings


def get_generator(generator_type: str):
    """Factory function to create the appropriate generator. Lazy import to avoid loading models at startup."""
    if generator_type == "gpt2":
        from .models.local.gpt2 import GPT2Generator
        return GPT2Generator()
    elif generator_type == "gemini":
        from .models.api.gemini import GeminiGenerator
        return GeminiGenerator()
    elif generator_type == "groq":
        from .models.api.groq import GroqGenerator
        return GroqGenerator()
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def build_index(limit: int | None):
    from .data.loader import load_wikipedia_simple
    from .data.preprocessor import simple_chunk, clean_text
    from .vectordb.chroma_client import ChromaIndexer
    
    ds = load_wikipedia_simple(limit=limit)
    print(f"Loaded articles: {len(ds)}")

    indexer = ChromaIndexer(settings.chroma_path, settings.collection_name, settings.embedding_model)
    indexer.create_collection(recreate=True)

    total_chunks = 0
    for doc_id, item in enumerate(ds):
        title = clean_text(item.get("title", ""))
        text = item.get("text", "")

        chunks = simple_chunk(text, chunk_size=settings.chunk_size, overlap=settings.overlap)
        if not chunks:
            continue

        indexer.add_chunks(doc_id, title, chunks)
        total_chunks += len(chunks)

    print(f"Done. Uploaded chunks: {total_chunks}")


# Run RAG based chat
# top_k - how many relevant chunks to retrieve for each query
# generator_type - generator type - local/api provider (gpt2/gemini/groq)
def run_chat(top_k: int | None = None, generator_type: str | None = None, 
             use_reranking: bool = False):
    from .vectordb.chroma_client import ChromaIndexer
    from .rag.pipeline import RAGPipeline
    
    indexer = ChromaIndexer(settings.chroma_path, settings.collection_name)
    indexer.get_collection()
    
    actual_top_k = top_k or settings.top_k
    
    # Use provided generator or default from settings
    gen_type = generator_type or settings.generator_type
    generator = get_generator(gen_type)
    print(f"Using generator: {gen_type}")
    print(f"Retrieval settings: top_k={actual_top_k}, max_distance={settings.max_distance}\n")
    if use_reranking:
        print(f"reranking=enabled\n")
    else:
        print()

    pipeline = RAGPipeline(indexer, generator, top_k=actual_top_k, 
                          use_reranking=use_reranking)

    print("Type 'exit' to stop.\n")
    while True:
        query = input("Prompt: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        answer, titles = pipeline.answer(query)
        print(f"\nAnswer: {answer}")
        print(f"\n==========\n")
        print(f"Sources: {', '.join(dict.fromkeys(titles))}\n")



def main():
    parser = argparse.ArgumentParser(
        description="RAG System with Wikipedia SimpleEnglish dataset",
        epilog="Example: python -m src.main --mode chat --generator groq --top_k 5"
    )
    parser.add_argument(
        "--mode", 
        choices=["index", "chat", "bench", "all"], 
        default=None,
        help="Mode: index (build VectorDB), chat (interactive), bench (speed test), all (index+chat)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=settings.limit,
        help=f"Number of articles to index (default: {'all articles' if settings.limit is None else settings.limit})"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=None,
        help=f"Number of chunks to retrieve for each query (default: {settings.top_k})"
    )
    parser.add_argument(
        "--generator",
        choices=["gpt2", "gemini", "groq"],
        default=None,
        help="Generator: gpt2 (local), gemini (API), groq (API). Default: from settings"
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=10, 
        help="Number of benchmark runs (default: 10)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable CrossEncoder reranking for better relevance (slower but more accurate)"
    )
    args = parser.parse_args()

    # If no mode specified, show help
    if args.mode is None:
        parser.print_help()
        return

    if args.mode in ("index", "all"):
        build_index(limit=args.limit)

    if args.mode == "bench":
        run_benchmark(top_k=args.top_k, generator_type=args.generator, runs=args.runs)

    if args.mode in ("chat", "all"):
        run_chat(top_k=args.top_k, generator_type=args.generator, 
                use_reranking=args.rerank)


if __name__ == "__main__":
    main()