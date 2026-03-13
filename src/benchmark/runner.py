import time
from pathlib import Path
from ..config.settings import settings

DEFAULT_RUNS_PER_QUERY = 10
QUESTIONS_FILE = Path("data/benchmark_questions.txt")

def get_test_queries() -> list[str]:
    """Load benchmark questions."""
    if QUESTIONS_FILE.exists():
        lines = [line.strip() for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            return lines
    else:
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")

def _build_context(indexer, query: str, top_k: int) -> tuple[str, float]:
    """Perfrom retrieval part and return retrieved context with retrieval time in ms."""
    t0 = time.perf_counter()
    res = indexer.search(query, top_k=top_k)
    docs = res["documents"][0]
    dists = res["distances"][0]
    filtered_docs = [d for d, dist in zip(docs, dists) if dist <= settings.max_distance]

    if filtered_docs:
        context = "\n\n".join(d[:settings.context_chars_per_chunk] for d in filtered_docs[:top_k])
    else:
        context = ""

    retrieval_time_ms = (time.perf_counter() - t0) * 1000
    return context, retrieval_time_ms


def run_benchmark(
    top_k: int | None = None,
    generator_type: str | None = None,
    runs_per_query: int = DEFAULT_RUNS_PER_QUERY,
):
    """Benchmark retrieval and generation speed."""
    from ..vectordb.chroma_client import ChromaIndexer
    from ..models.factory import get_generator

    indexer = ChromaIndexer(settings.chroma_path, settings.collection_name)
    indexer.get_collection()

    actual_top_k = top_k or settings.top_k
    gen_type = generator_type or settings.generator_type
    generator = get_generator(gen_type)
    test_queries = get_test_queries()

    retrieval_times = []
    generation_times = []
    total_times = []

    print(f"Using generator: {gen_type}")
    print(f"Questions: {len(test_queries)}")
    print(f"Runs per question: {runs_per_query}")
    print()

    # One lightweight warmup (not measured)
    warmup_prompt = ("Say: I don't know.")
    _ = generator.generate(warmup_prompt, max_tokens=10)

    print("Warmup done.\n")

    total_runs = len(test_queries) * runs_per_query
    run_idx = 0

    for query in test_queries:
        for _ in range(runs_per_query):
            run_idx += 1

            t0 = time.perf_counter()
            context, retrieval_time = _build_context(indexer, query, actual_top_k)
            t1 = time.perf_counter()

            if context:
                prompt = (
                    "Use ONLY the context. If not enough info, say: I don't know.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {query}\n"
                    "Answer:"
                )
                _ = generator.generate(prompt, max_tokens=60)

            t2 = time.perf_counter()
            generation_time = (t2 - t1) * 1000  # ms
            total_time = (t2 - t0) * 1000  # ms

            retrieval_times.append(retrieval_time)
            generation_times.append(generation_time)
            total_times.append(total_time)

            print(
                f"Run {run_idx}/{total_runs} | query='{query[:45]}': "
                f"retrieval={retrieval_time:.2f}ms, generation={generation_time:.2f}ms, total={total_time:.2f}ms"
            )

    stats_retrieval = retrieval_times
    stats_generation = generation_times
    stats_total = total_times

    avg_retrieval = sum(stats_retrieval) / len(stats_retrieval)
    avg_generation = sum(stats_generation) / len(stats_generation)
    avg_total = sum(stats_total) / len(stats_total)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Generator: {gen_type}")
    print("Questions file: data/benchmark_questions.txt (or built-in defaults)")
    print("Warmup: 1 pass")
    print(f"Total Measured Runs: {len(stats_total)}")
    print(f"Top-k: {actual_top_k}")
    print(f"Max distance: {settings.max_distance}")
    print("-" * 70)
    print(f"Avg Retrieval Time:  {avg_retrieval:.2f} ms")
    print(f"Avg Generation Time: {avg_generation:.2f} ms")
    print(f"Avg Total Time:      {avg_total:.2f} ms")
    print("=" * 70)
