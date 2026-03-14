from ..vectordb.chroma_client import ChromaIndexer
from ..models.base import Generator
from ..config.settings import settings


class RAGPipeline:
    def __init__(self, indexer: ChromaIndexer, generator: Generator, top_k: int | None = None, 
                 max_distance: float | None = None, context_chars: int | None = None,
                 use_reranking: bool | None = None):
        self.indexer = indexer
        self.generator = generator
        self.top_k = top_k or settings.top_k
        self.max_distance = max_distance or settings.max_distance
        self.context_chars = context_chars or settings.context_chars_per_chunk
        self.small_to_big_enabled = settings.small_to_big_enabled
        self.small_to_big_window = settings.small_to_big_window
        
        # Advanced retrieval options
        self.use_reranking = use_reranking if use_reranking is not None else settings.use_reranking
        
        # Lazy initialization of reranker
        self._reranker = None
    
    @property
    def reranker(self):
        """Lazy load reranker"""
        if self._reranker is None and self.use_reranking:
            from ..retrieval.reranker import Reranker
            self._reranker = Reranker(model_name=settings.reranker_model)
        return self._reranker

    def answer(self, query: str) -> tuple[str, list[str]]:
        """
        Generate answer for a query using RAG pipeline.
        """
        # Initial retrieval from vector database
        # Retrieve more docs if reranking is enabled
        initial_top_k = self.top_k * 2 if self.use_reranking else self.top_k
        res = self.indexer.search(query, top_k=initial_top_k)
        
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        
        if not docs:
            return "I don't have enough information to answer this question.", []
        
        # Apply reranking if enabled
        if self.use_reranking and self.reranker:
            docs, metas, rerank_scores = self.reranker.rerank(
                query, docs, metas, dists, top_k=settings.rerank_top_k
            )
            # Update distances with rerank scores for filtering
            dists = [(1 - score) for score in rerank_scores]  # Convert to distance-like metric
        
        # Filter by distance threshold
        filtered_docs = []
        filtered_metas = []
        filtered_dists = []
        for d, m, dist in zip(docs, metas, dists):
            if dist <= self.max_distance:
                filtered_docs.append(d)
                filtered_metas.append(m)
                filtered_dists.append(dist)
        
        if not filtered_docs:
            return "I don't have enough information to answer this question.", []
        
        items = list(zip(filtered_docs, filtered_metas, filtered_dists))

        items = items[:self.top_k]

        context_parts = []
        used_ids = set()
        for doc, meta, _ in items:
            doc_id = meta.get("doc_id") if isinstance(meta, dict) else None
            chunk_id = meta.get("chunk_id") if isinstance(meta, dict) else None

            key = (doc_id, chunk_id)
            if key in used_ids:
                continue
            used_ids.add(key)

            if (
                self.small_to_big_enabled
                and isinstance(doc_id, int)
                and isinstance(chunk_id, int)
            ):
                window_docs = self.indexer.get_chunk_window(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    window=self.small_to_big_window,
                )
                if window_docs:
                    expanded = "\n".join(window_docs)
                    context_parts.append(expanded[:self.context_chars])
                    continue

            context_parts.append(doc[:self.context_chars])

        # Build context and generate prompt from template
        context = "\n\n".join(context_parts)
        prompt = settings.prompt_template.format(context=context, query=query)
                
        # Generate answer using the selected generator
        answer = self.generator.generate(prompt, max_tokens=60)
        
        # Extract titles from metadata for sources
        titles = [m.get("title", "unknown") for _, m, _ in items]
        
        return answer, titles
