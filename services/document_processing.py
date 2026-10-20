import os
from datetime import datetime, timezone

from RAG_Engine.contracts.interfaces import (
    DocumentProcessingInterface,
    DocumentParserInterface,
    DocumentChunkerInterface,
    LocalTempStorageInterface,
    EmbedderInterface,
    VectorStoreInterface,
)
from RAG_Engine.infrastructure.storage.storage_quota import StorageQuotaManager


class DocumentProcessing(DocumentProcessingInterface):

    def __init__(
        self,
        parser: DocumentParserInterface,
        chunker: DocumentChunkerInterface,
        temp_storage: LocalTempStorageInterface,
        embedder: EmbedderInterface,
        vector_store: VectorStoreInterface,
        quota_manager: StorageQuotaManager | None = None,
        embedding_dim: int = 3072, # Valor por defecto 
        batch_size: int = 512      # Valor por defecto
    ):
        self.parser = parser
        self.chunker = chunker
        self.temp_storage = temp_storage
        self.embedder = embedder
        self.vector_store = vector_store
        self.quota_manager = quota_manager or StorageQuotaManager(vector_store)
        self.embedding_dim = embedding_dim 
        self.batch_size = batch_size       

    # =========================================================
    # INDEX
    # =========================================================

    def index(self, file_bytes: bytes, file_name: str, user_id: str) -> dict:

        if self.vector_store.file_exists(file_name):
            raise ValueError(f"Ya existe un archivo con el nombre '{file_name}'.")

        suffix    = "." + file_name.split(".")[-1]
        temp_path = self.temp_storage.save(file_bytes, suffix)

        try:
            # ── 1) Parsing ─────────────────────────────────
            try:
                parsed_document = self.parser.parse(temp_path)
            except Exception as e:
                raise ValueError(f"Error durante el parsing: {e}")

            # ── 2) Chunking ────────────────────────────────
            try:
                chunks = self.chunker.chunk(parsed_document, file_name)
            except Exception as e:
                raise ValueError(f"Error durante el chunking: {e}")

            # ── 3) Quota check ─────────────────────────────
            quota_result = self.quota_manager.check(
                n_chunks=len(chunks),
                embedding_dim=self.embedding_dim, 
            )

            if not quota_result.allowed:
                raise ValueError(quota_result.warning_message)

            # ── 4) Build docs ──────────────────────────────
            created_at = datetime.now(timezone.utc).isoformat()
            docs = []
            metadata = []

            for chunk in chunks:
                text = chunk["text"]
                meta = chunk.get("metadata", {}).copy()
                meta["user_id"]    = user_id
                meta["created_at"] = created_at
                docs.append(text)
                metadata.append(meta)

            batch_size = 512
            session_count = 0

            # ── 5) Embeddings + upsert ─────────────────────
            try:
                for start in range(0, len(docs), self.batch_size):
                    end = start + self.batch_size

                    batch_docs = docs[start:end]
                    batch_meta = metadata[start:end]
                    batch_ids  = [f"{created_at}_{start+i}" for i in range(len(batch_docs))]

                    embeddings = self.embedder.generate_embeddings(batch_docs)

                    self.vector_store.upsert(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_meta,
                        embeddings=embeddings.tolist(),
                    )

                    session_count += len(batch_docs)

            except Exception as e:
                raise ValueError(f"Error generando embeddings o guardando: {e}")

            # ── 6) File registry ───────────────────────────
            try:
                self.vector_store.register_file(
                    user_id=user_id,
                    file_name=file_name,
                    embeddings_count=session_count,
                )
            except Exception as e:
                raise ValueError(f"Error registrando archivo: {e}")

            return {
                "file_name": file_name,
                "embeddings_created": session_count,
                "status": "success",
                "storage_estimate": quota_result.estimate.summary(),
            }

        finally:
            self.temp_storage.delete(temp_path)

    # =========================================================
    # DELETE (user)
    # =========================================================

    def delete(self, file_name: str, user_id: str):

        if not self.vector_store.file_exists(file_name):
            raise ValueError("El archivo no existe")

        if not self.vector_store.is_owner(user_id, file_name):
            raise ValueError("No tiene permisos")

        ids = self.vector_store.find_ids({
            "$and": [
                {"source": file_name},
                {"user_id": user_id},
            ]
        })

        if not ids:
            raise ValueError("No hay embeddings asociados")

        self.vector_store.delete_by_ids(ids)
        self.vector_store.remove_file_record_by_id(user_id, file_name)

        return {"deleted": len(ids)}

    # =========================================================
    # DELETE (admin)
    # =========================================================

    def delete_as_admin(self, file_name: str):

        if not self.vector_store.file_exists(file_name):
            raise ValueError("El archivo no existe")

        ids = self.vector_store.find_ids({"source": file_name})

        if not ids:
            raise ValueError("No hay embeddings asociados")

        self.vector_store.delete_by_ids(ids)
        self.vector_store.remove_file_record(file_name)

        return {"deleted": len(ids)}
    
    # =========================================================
    # LIST
    # =========================================================

    def list_documents(self):
        return self.vector_store.list_files()
    


