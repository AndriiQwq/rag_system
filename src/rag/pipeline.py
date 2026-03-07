from ..vectordb.chroma_client import ChromaIndexer
from ..models.base import Generator


class RAGPipeline:
    def __init__(self, indexer: ChromaIndexer, generator: Generator, top_k: int = 3):
        self.indexer = indexer
        self.generator = generator
        self.top_k = top_k

    def answer(self, query: str) -> tuple[str, list[str]]:
        # Search for relevant chunks in the vector database
        res = self.indexer.search(query, top_k=self.top_k)
        
        docs = res["documents"][0] # retrieved chunks 
        metas = res["metadatas"][0] # metadata for retrieved chunks
        dists = res["distances"][0] # distances (similarity scores) for retrieved chunks
        
        # Retrieve only the relevant chunks based on distance threshold
        # For cosine distance: lower is better. 1.2 is more permissive than 1.0
        filtered_docs = [d for d, dist in zip(docs, dists) if dist <= 1.2]
        if not filtered_docs:
            return "I don't know based on retrieved context.", []
        
        # Take up to 3 chunks, 600 chars each = ~1800 chars context
        context = "\n\n".join(d[:600] for d in filtered_docs[:3])
        
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
