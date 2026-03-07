import argparse
import time

from .data.loader import load_wikipedia_simple
from .data.preprocessor import simple_chunk, clean_text
from .config.settings import settings
from .vectordb.chroma_client import ChromaIndexer
from .rag.pipeline import RAGPipeline
from .models.local.gpt2 import GPT2Generator
from .models.api.gemini import GeminiGenerator
from .models.api.groq import GroqGenerator


def get_generator(generator_type: str):
    """Factory function to create the appropriate generator."""
    if generator_type == "gpt2":
        return GPT2Generator()
    elif generator_type == "gemini":
        return GeminiGenerator()
    elif generator_type == "groq":
        return GroqGenerator()
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def build_index(limit: int | None):
    ds = load_wikipedia_simple(limit=limit)
    print(f"Loaded articles: {len(ds)}")

    indexer = ChromaIndexer(settings.chroma_path, settings.collection_name)
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
def run_chat(top_k: int = 3, generator_type: str | None = None):
    indexer = ChromaIndexer(settings.chroma_path, settings.collection_name)
    indexer.get_collection()
    
    # Use provided generator or default from settings
    gen_type = generator_type or settings.generator_type
    generator = get_generator(gen_type)
    print(f"Using generator: {gen_type}\n")

    pipeline = RAGPipeline(indexer, generator, top_k=top_k)

    print("Type 'exit' to stop.\n")
    while True:
        query = input("Prompt: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        answer, titles = pipeline.answer(query)
        print(f"\nAnswer: {answer}")
        print(f"Sources: {', '.join(dict.fromkeys(titles))}\n")



def main():
    parser = argparse.ArgumentParser(
        description="RAG System with Wikipedia SimpleEnglish dataset",
        epilog="Example: python -m src.main --mode chat --generator groq --top_k 5"
    )
    parser.add_argument(
        "--mode", 
        choices=["index", "chat", "bench", "all"], 
        default="chat",
        help="Mode: index (build VectorDB), chat (interactive), bench (speed test), all (index+chat)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=settings.limit if settings.limit is not None else 1000,
        help="Number of articles to index (default: 1000)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=3,
        help="Number of chunks to retrieve for each query (default: 3)"
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
    args = parser.parse_args()

    if args.mode in ("index", "all"):
        build_index(limit=args.limit)

    if args.mode == "bench":
        run_benchmark(top_k=args.top_k, generator_type=args.generator, runs=args.runs)

    if args.mode in ("chat", "all"):
        run_chat(top_k=args.top_k, generator_type=args.generator)


if __name__ == "__main__":
    main()