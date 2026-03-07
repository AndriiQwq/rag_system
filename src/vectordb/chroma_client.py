import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


class ChromaIndexer:
    def __init__(self, chroma_path: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection_name = collection_name
        self.collection = None
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def create_collection(self, recreate: bool = True):
        if recreate:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
        
        self.collection = self.client.create_collection(name=self.collection_name)

    def get_collection(self):
        self.collection = self.client.get_collection(self.collection_name)

    def add_chunks(self, doc_id: int, title: str, chunks: List[str]):
        vectors = self.embedder.encode(chunks, normalize_embeddings=True).tolist()
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metas = [{"title": title, "doc_id": doc_id, "chunk_id": i} for i in range(len(chunks))]
        
        self.collection.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metas)

    def search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        qvec = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        return self.collection.query(query_embeddings=[qvec], n_results=top_k)