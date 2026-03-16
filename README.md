# How to start

```bash
python -m venv .venv

.venv\Scripts\activate
# or 
source .venv/bin/activate

pip install -r requirements.txt

kaggle datasets download -d ffatty/plain-text-wikipedia-simpleenglish -p data --unzip
```

---

## Overview

The documents from Wikipedia were indexed and split into chunks.
To build better chunks, a token-based sentence-aware method was used to keep information coherent inside each chunk and keep chunk size suitable for the model.

Tests on the full document set showed the following results.

GPT-2 is a relatively weak model, so it is important to choose parameters carefully to get stable output quality.
The model input limit is 512 tokens.

## Retrieval

Retrieval settings used:

```python
top_k: int = 4
max_distance: float = 1.3
context_chars_per_chunk: int = 700
small_to_big_enabled: bool = False
use_reranking: bool = True
small_to_big_window: int = 1
rerank_top_k: int = 4
```

CrossEncoder reranking was also used with:

- cross-encoder/ms-marco-MiniLM-L-6-v2

The vector database is local ChromaDB, with embeddings model:

- BAAI/bge-large-en-v1.5


For experiments, Gemini and Groq API models were also tested.
They can be used for automatic answer-quality evaluation of GPT-2 outputs.

## GPT-2 settings

Because this task is focused on GPT-2, the following generation parameters were selected:

```python
gpt2_max_new_tokens: int = 20
gpt2_do_sample: bool = False
gpt2_temperature: float = 0.6
gpt2_top_p: float = 0.8
gpt2_top_k: int | None = 40
gpt2_no_repeat_ngram_size: int = 4
gpt2_max_input_length: int = 512
```

It is also important to tune the model decoding parameters, because they directly affect answer quality:

- `gpt2_max_new_tokens`
- `gpt2_do_sample`
- `gpt2_temperature`
- `gpt2_top_p` & `gpt2_top_k`
- `gpt2_no_repeat_ngram_size`

Tuning of GPT-2 generation parameters was conducted, and the best found setup for this project is:

- `gpt2_max_new_tokens = 20`
- `gpt2_do_sample = False`
- `gpt2_temperature = 0.6`
- `gpt2_top_p = 0.8`
- `gpt2_top_k = 40`
- `gpt2_no_repeat_ngram_size = 4`

Prompt is also important. In the latest comparison, a simpler prompt showed better quality.

```python
prompt_template: str = """Context: {context}

Question: {query}
Answer:"""
```

## CLI usage

The project can be run as a standard CLI application.

Options:

```text
-h, --help            show this help message and exit
--mode {index,chat,bench}
                      Mode: index (build VectorDB), chat (interactive Q&A),
                      bench (benchmark latency+quality)
--limit LIMIT         Number of articles to index (default: all articles)
--top_k TOP_K         Number of chunks to retrieve for each query (default: 3)
--generator {gpt2,gemini,groq}
                      Generator: gpt2 (local), gemini (API), groq (API)
--runs RUNS           Number of benchmark runs (default: 3)
--rerank              Enable CrossEncoder reranking (slower but potentially more accurate)
--wipe-db             Delete local Chroma DB folder before indexing (full clean rebuild)
```

Example:

```bash
python -m src.main --mode chat
```

## Benchmark summary

Stage 1 benchmark on 4 combinations showed the best results with:

- `rerank = True` (similar quality with and without `small_to_big`)

Base setup:

- Queries used: 10
- Generator: gpt2
- Runs per query: 3
- Base params: top_k=3, max_distance=1.2, context_chars=500

Stage 2 tuning was done with:

- top_k: 2 / 3 / 5
- max_distance: 1.0 / 1.2 / 1.5
- context_chars: 200 / 300 / 500

Best Stage 2 rows:

```text
config_id,phase,label,small_to_big,rerank,top_k,max_distance,context_chars,queries,runs_per_query,avg_total_ms,avg_answer_len,empty_answer_rate,avg_judge_score
w1-C8 ... s2b=False rr=True top_k=3 dist=1.2 ctx=300 ... avg_total_ms=1264.6 ... avg_judge_score=2.6
w1-C9 ... s2b=False rr=True top_k=3 dist=1.2 ctx=500 ... avg_total_ms=1224.88 ... avg_judge_score=2.6
```

Best Stage 3 rows:

```text
config_id,label,max_new_tokens,do_sample,small_to_big,rerank,top_k,max_distance,context_chars,queries,runs_per_query,avg_total_ms,avg_answer_len,empty_answer_rate,avg_judge_score
g1,"max_new_tokens=20, do_sample=False",20,False,False,True,3,1.2,500,10,1,499.59,78.2,0.0,2.5
g2,"max_new_tokens=20, do_sample=True",20,True,False,True,3,1.2,500,10,1,494.73,80.4,0.0,2.8
g3,"max_new_tokens=30, do_sample=False",30,False,False,True,3,1.2,500,10,1,658.27,125.1,0.0,3.3
```

Final selected config for Stage 3 (best quality/speed balance):

```python
# Retrieval
top_k: int = 3
max_distance: float = 1.2
context_chars_per_chunk: int = 500
small_to_big_enabled: bool = False
use_reranking: bool = True

# Generation
gpt2_max_new_tokens: int = 30
gpt2_do_sample: bool = False
```

Benchmark outputs are stored in data/ directory. The quality of the GPT-2-generated text was evaluated using an LLM.
After tuning, answer quality improved, but GPT-2 still makes factual and reasoning errors in some cases.
This is expected and related to the model capacity/quality level of GPT-2.

### Timing benchmark

Benchmarks performed on `max_new_tokens` were run on 10 curated questions, 3 runs per query (30 total runs per config).

| Config | max\_new\_tokens | do\_sample | Retrieval ms | Generation ms | Total ms | Score / 5 |
|--------|-----------------|------------|-------------:|--------------:|---------:|----------:|
| sp1    | **20**          | False      | 65           | 394           | **459**  | 1.00      |
| sp2    | 30              | False      | 61           | 596           | 657      | 3.00      |
| sp3    | 40              | False      | 61           | 778           | 838      | **4.25**  |

Quality analysis was done with LLM-based scoring due to limited project time.
