import chromadb
from chromadb.config import Settings
from .config import CHROMA_DIR

class Retriever:
    def __init__(self, collection_name="gitlab_docs"):
        self.client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
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
        self.client.persist()
