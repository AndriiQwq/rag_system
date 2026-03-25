import time
from pathlib import Path
from ..config.settings import settings

DEFAULT_RUNS_PER_QUERY = 3
QUESTIONS_FILE = Path("data/benchmark/benchmark_questions.txt")

def get_test_queries() -> list[str]:
    """Load benchmark questions from file."""
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")
    lines = [line.strip() for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Questions file is empty: {QUESTIONS_FILE}")
    return lines


def _is_empty_answer(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return True
    return (
        "i don't have enough information" in text
        or "i don't know" in text
    )


def run_benchmark(
    top_k: int | None = None,
    generator_type: str | None = None,
    runs_per_query: int = DEFAULT_RUNS_PER_QUERY,
):
    """Benchmark full RAG pipeline latency (same path as chat mode)."""
    from ..vectordb.chroma_client import ChromaIndexer
    from ..models.factory import get_generator
    from ..rag.pipeline import RAGPipeline

    indexer = ChromaIndexer(settings.chroma_path, settings.collection_name)
    indexer.get_collection()

    actual_top_k = top_k or settings.top_k
    gen_type = generator_type or settings.generator_type
    generator = get_generator(gen_type)
    test_queries = get_test_queries()
    pipeline = RAGPipeline(
        indexer=indexer,
        generator=generator,
        top_k=actual_top_k,
        max_distance=settings.max_distance,
        context_chars=settings.context_chars_per_chunk,
        use_reranking=settings.use_reranking,
    )

    total_times = []
    answer_lengths = []
    empty_answers = 0

    print(f"Using generator: {gen_type}")
    print(f"Questions: {len(test_queries)}")
    print(f"Runs per question: {runs_per_query}")
    print()

    # One lightweight warmup (not measured)
    _ = pipeline.answer(".")

    print("Warmup done.\n")

    total_runs = len(test_queries) * runs_per_query
    run_idx = 0

    for query in test_queries:
        for _ in range(runs_per_query):
            run_idx += 1

            t0 = time.perf_counter()
            answer, _sources = pipeline.answer(query)
            t1 = time.perf_counter()
            total_time = (t1 - t0) * 1000  # ms

            answer_len = len((answer or "").strip())
            answer_lengths.append(answer_len)
            if _is_empty_answer(answer):
                empty_answers += 1
            total_times.append(total_time)

            print(
                f"Run {run_idx}/{total_runs} | query='{query[:45]}' | "
                f"total={total_time:.2f}ms | answer_len={answer_len}"
            )

    stats_total = total_times

    avg_total = sum(stats_total) / len(stats_total)
    avg_answer_len = sum(answer_lengths) / len(answer_lengths)
    empty_rate = empty_answers / len(stats_total)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Generator: {gen_type}")
    print("Questions file: data/benchmark_questions.txt")
    print(f"Total Measured Runs: {len(stats_total)}")
    print(f"Top-k: {actual_top_k}")
    print(f"Max distance: {settings.max_distance}")
    print(f"Reranking: {settings.use_reranking}")
    print("-" * 70)
    print(f"Avg Total Time:      {avg_total:.2f} ms")
    print(f"Avg Answer Length:   {avg_answer_len:.2f} chars")
    print(f"Empty Answer Rate:   {empty_rate:.2%}")
    print("=" * 70)
