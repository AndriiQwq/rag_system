import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


class ChromaIndexer:
    def __init__(self, chroma_path: str, collection_name: str, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection_name = collection_name
        self.collection = None
        self.embedder = SentenceTransformer(embedding_model)

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

    def get_chunk_window(self, doc_id: int, chunk_id: int, window: int = 1) -> List[str]:
        """Returns neighboring chunks around the selected chunk."""
        if self.collection is None:
            self.get_collection()

        start_chunk = max(0, chunk_id - window)
        end_chunk = chunk_id + window
        ids = [f"{doc_id}_{i}" for i in range(start_chunk, end_chunk + 1)]

        res = self.collection.get(ids=ids, include=["documents", "metadatas"])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])

        pairs = []
        for doc, meta in zip(docs, metas):
            current_chunk_id = meta.get("chunk_id", 10**9) if isinstance(meta, dict) else 10**9
            pairs.append((current_chunk_id, doc))

        pairs.sort(key=lambda x: x[0])
        return [doc for _, doc in pairs]