from ..config.settings import settings


# RAG based chat
# top_k - how many relevant chunks to retrieve for each query
# generator_type - generator type - local/api provider (gpt2/gemini/groq)
def run_chat(top_k: int | None = None, generator_type: str | None = None,
             use_reranking: bool = False):
    from ..vectordb.chroma_client import ChromaIndexer
    from ..rag.pipeline import RAGPipeline
    from ..models.factory import get_generator

    indexer = ChromaIndexer(settings.chroma_path, settings.collection_name)
    indexer.get_collection()

    actual_top_k = top_k or settings.top_k

    # Use provided generator or default from settings
    gen_type = generator_type or settings.generator_type
    generator = get_generator(gen_type)
    print(f"Using generator: {gen_type}")
    print(f"Retrieval settings: top_k={actual_top_k}, max_distance={settings.max_distance}\n")
    if use_reranking:
        print("reranking=enabled\n")
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
        print("\n==========\n")
        print(f"Sources: {', '.join(dict.fromkeys(titles))}\n")
