import argparse
import shutil
from pathlib import Path

from .config.settings import settings
from .cli.chat import run_chat
from .benchmark.runner import run_benchmark


def build_index(limit: int | None, wipe_db: bool = False):
    from .data.loader import load_wikipedia_simple
    from .data.preprocessor import clean_text, hybrid_chunk, simple_chunk
    from .vectordb.chroma_client import ChromaIndexer
    from tqdm.auto import tqdm

    if wipe_db:
        db_path = Path(settings.chroma_path)
        if db_path.exists():
            shutil.rmtree(db_path)
            print(f"Removed existing DB folder: {db_path}")
    
    ds = load_wikipedia_simple(limit=limit)
    print(f"Loaded articles: {len(ds)}")

    print("Chunking strategy:", settings.chunking_strategy)
    if settings.chunking_strategy == "hybrid":
        print(
            "Chunking params:",
            {
                "max_tokens": settings.chunk_max_tokens,
                "overlap_sentences": settings.chunk_overlap_sentences,
                "tokenizer": settings.chunk_tokenizer_name,
                "long_sentence_overlap_tokens": settings.long_sentence_overlap_tokens,
            },
        )
    else:
        print(
            "Chunking params:",
            {
                "chunk_size": settings.chunk_size,
                "overlap": settings.overlap,
            },
        )

    indexer = ChromaIndexer(settings.chroma_path, settings.collection_name, settings.embedding_model)
    indexer.create_collection(recreate=True)

    total_chunks = 0
    progress = tqdm(
        enumerate(ds),
        total=len(ds),
        desc=f"Indexing [{settings.chunking_strategy}]",
        unit="article",
    )
    for doc_id, item in progress:
        title = clean_text(item.get("title", ""))
        text = item.get("text", "")

        if settings.chunking_strategy == "hybrid":
            chunks = hybrid_chunk(
                text,
                max_tokens=settings.chunk_max_tokens,
                overlap_sentences=settings.chunk_overlap_sentences,
                tokenizer_name=settings.chunk_tokenizer_name,
                long_sentence_overlap_tokens=settings.long_sentence_overlap_tokens,
            )
        else:
            chunks = simple_chunk(text, chunk_size=settings.chunk_size, overlap=settings.overlap)
        
        if not chunks: # For ex. empty articles
            continue

        indexer.add_chunks(doc_id, title, chunks)
        total_chunks += len(chunks)
        progress.set_postfix(chunks=total_chunks)

    print(f"Done. Uploaded chunks: {total_chunks}")
    
def main():
    parser = argparse.ArgumentParser(
        description="RAG System with Wikipedia SimpleEnglish dataset",
        epilog="Example: python -m src.main --mode chat --generator gpt2"
    )
    parser.add_argument(
        "--mode", 
        choices=["index", "chat", "bench"], 
        default=None,
        help="Mode: index (build VectorDB), chat (interactive Q&A), bench (benchmark latency+quality)"
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
        default=3, 
        help="Number of benchmark runs per query (default: 3)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable CrossEncoder reranking for better relevance (slower but more accurate)"
    )
    parser.add_argument(
        "--wipe-db",
        action="store_true",
        help="Delete local Chroma DB folder before indexing (full clean rebuild)"
    )
    args = parser.parse_args()

    # If no mode specified, show help
    if args.mode is None:
        parser.print_help()
        return

    if args.mode == "index":
        build_index(limit=args.limit, wipe_db=args.wipe_db)

    if args.mode == "bench":
        run_benchmark(
            top_k=args.top_k,
            generator_type=args.generator,
            runs_per_query=args.runs,
        )

    if args.mode == "chat":
        run_chat(top_k=args.top_k, generator_type=args.generator,
                use_reranking=args.rerank)


if __name__ == "__main__":
    main()