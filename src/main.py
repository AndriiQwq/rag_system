import argparse

from .config.settings import settings
from .cli.chat import run_chat
from .benchmark.runner import run_benchmark


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