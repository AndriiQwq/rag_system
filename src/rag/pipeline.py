from ..vectordb.chroma_client import ChromaIndexer
from ..models.base import Generator
from ..config.settings import settings


class RAGPipeline:
    def __init__(self, indexer: ChromaIndexer, generator: Generator, top_k: int | None = None, 
                 max_distance: float | None = None, context_chars: int | None = None):
        self.indexer = indexer
        self.generator = generator
        self.top_k = top_k or settings.top_k
        self.max_distance = max_distance or settings.max_distance
        self.context_chars = context_chars or settings.context_chars_per_chunk

    def answer(self, query: str) -> tuple[str, list[str]]:
        # Search for relevant chunks in the vector database
        res = self.indexer.search(query, top_k=self.top_k)
        
        docs = res["documents"][0] # retrieved chunks 
        metas = res["metadatas"][0] # metadata for retrieved chunks
        dists = res["distances"][0] # distances (similarity scores) for retrieved chunks
        
        # Retrieve only the relevant chunks based on distance threshold
        # For cosine distance: lower is better (0=identical, higher=less similar)
        filtered_docs = [d for d, dist in zip(docs, dists) if dist <= self.max_distance]
        if not filtered_docs:
            return "I don't know based on retrieved context.", []
        
        # Build context from filtered chunks
        context = "\n\n".join(d[:self.context_chars] for d in filtered_docs)
        
        prompt = (
            "Use ONLY the context. If not enough info, say: I don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        
        # Generate answer using the selected generator
        answer = self.generator.generate(prompt, max_tokens=120)
        
        # Extract titles from metadata for sources
        titles = [m.get("title", "unknown") for m in metas]
        
        return answer, titles
