import chromadb
from .config import CHROMA_DIR

class Retriever:
    def __init__(self, collection_name="gitlab_docs"):
        # Use PersistentClient for ChromaDB 0.5.x
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.col = self.client.get_or_create_collection(collection_name)

    def query(self, query_emb, top_k=4):
        res = self.col.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return res

    def add(self, ids, docs, metas, embs):
        self.col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        # Note: In newer ChromaDB versions, persistence is automatic
        # self.client.persist() is no longer needed
