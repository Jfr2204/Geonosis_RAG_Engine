import chromadb
from RAG_Engine.contracts.interfaces import VectorStoreInterface

from pathlib import Path

class ChromaVectorStore(VectorStoreInterface):

    def __init__(self, db_path: str):

        if not db_path:
            raise ValueError("CHROMA_DB_PATH no definido")

        self.client = chromadb.PersistentClient(path=db_path)

        # ==============================
        # COLLECTION 1 - EMBEDDINGS
        # ==============================

        self.collection = self.client.get_or_create_collection(
            name="test_collection_pca",
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 200
                }
            }
        )

        # ==============================
        # COLLECTION 2 - FILE REGISTRY
        # ==============================

        self.file_registry = self.client.get_or_create_collection(
            name="file_registry"
        )

    # =====================================================
    # BASIC OPERATIONS (EMBEDDINGS)
    # =====================================================

    def upsert(self, ids, documents, metadatas, embeddings):
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def query(self, query_embeddings, n_results: int):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

    def delete_by_ids(self, ids: list[str]):
        if ids:
            self.collection.delete(ids=ids)

    def count(self):
        return self.collection.count()

    # =====================================================
    # GENERIC FIND
    # =====================================================

    def find_ids(self, where_filter: dict) -> list[str]:

        result = self.collection.get(
            where=where_filter
        )

        return result.get("ids", [])

    # =====================================================
    # FILE REGISTRY MANAGEMENT
    # =====================================================

    def _build_file_id(self, user_id: str, file_name: str) -> str:
        return f"{user_id}::{file_name}"

    def register_file(self, user_id: str, file_name: str, embeddings_count: int):

        file_id = self._build_file_id(user_id, file_name)

        self.file_registry.upsert(
            ids=[file_id],
            documents=[file_name],
            metadatas=[{
                "user_id": user_id,
                "file_name": file_name,
                "embeddings_count": embeddings_count
            }]
        )

    def file_exists(self, file_name: str) -> bool:

        result = self.file_registry.get(
            where={"file_name": file_name}
        )

        return len(result["ids"]) > 0

    def is_owner(self, user_id: str, file_name: str) -> bool:

        file_id = self._build_file_id(user_id, file_name)

        result = self.file_registry.get(ids=[file_id])

        return len(result["ids"]) > 0

    def list_files(self) -> list[dict]:

        result = self.file_registry.get(
            include=["metadatas"]
        )

        files = []

        for meta in result.get("metadatas", []):
            files.append({
                "file_name": meta.get("file_name"),
                "user_id": meta.get("user_id"),
                "embeddings_count": meta.get("embeddings_count")
            })

        return files

    def remove_file_record_by_id(self, user_id: str, file_name: str):

        file_id = self._build_file_id(user_id, file_name)
        self.file_registry.delete(ids=[file_id])

    def remove_file_record(self, file_name: str):

        result = self.file_registry.get(
            where={"file_name": file_name}
        )

        ids = result.get("ids", [])

        if ids:
            self.file_registry.delete(ids=ids)
