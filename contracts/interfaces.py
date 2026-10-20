from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import numpy as np

# =========================================================
# RAG ENGINE INTERFACE
# =========================================================

class RAGEngineInterface(ABC):
    """
    Interface for the RAG engine.

    Responsible for:
    - Document ingestion
    - Question answering
    - Document management
    """

    # =====================================================
    # DOCUMENT INGESTION
    # =====================================================

    @abstractmethod
    def load_document(
        self,
        file_bytes: bytes,
        file_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Index a document into the system.
        """
        pass

    # =====================================================
    # QUESTION ANSWERING
    # =====================================================

    @abstractmethod
    def ask(
        self,
        question: str,
        conversation_context: Optional[str],
        n_results: int
    ) -> List[str]:
        """
        Answer a user question using the RAG pipeline.
        """
        pass

    # =====================================================
    # DOCUMENT DELETION (USER)
    # =====================================================

    @abstractmethod
    def delete_document(
        self,
        file_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Delete a document owned by a specific user.
        """
        pass

    # =====================================================
    # DOCUMENT DELETION (ADMIN)
    # =====================================================

    @abstractmethod
    def delete_document_as_admin(
        self,
        file_name: str
    ) -> Dict[str, Any]:
        """
        Delete a document regardless of ownership.
        """
        pass

    # =====================================================
    # DOCUMENT LISTING
    # =====================================================

    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Return the list of indexed documents.
        """
        pass

# =========================================================
# DOCUMENT PROCESSING INTERFACE
# =========================================================

class DocumentProcessingInterface(ABC):
    """
    Interface for document processing.

    Responsible for:
    - Loading documents
    - Creating embeddings
    - Managing stored documents
    """

    @abstractmethod
    def index(
        self,
        file_bytes: bytes,
        file_name: str,
        user_id: str
    ) -> None:
        """
        Index a new document into the vector store.
        """
        pass

    @abstractmethod
    def delete(
        self,
        file_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Delete a document owned by a user.
        """
        pass

    @abstractmethod
    def delete_as_admin(
        self,
        file_name: str
    ) -> Dict[str, Any]:
        """
        Delete a document regardless of ownership.
        """
        pass

    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Return the list of indexed documents.
        """
        pass


# =========================================================
# QUESTION PROCESSING INTERFACE
# =========================================================

class QuestionProcessingInterface(ABC):
    """
    Interface for question answering.

    Responsible for:
    - Query generation
    - Retrieval
    - RAG response generation
    """

    @abstractmethod
    def answer(
        self,
        question: str,
        conversation_context: Optional[str],
        n_results: int
    ) -> List[str]:
        """
        Generate an answer for a user question using RAG.
        """
        pass


# =========================================================
# DOCUMENT PARSER INTERFACE
# =========================================================

class DocumentParserInterface(ABC):
    """
    Interface for document parsing.

    Responsible for:
    - Detecting file format
    - Extracting raw content from documents
    """

    @abstractmethod
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document and return a structured representation
        containing its extension and extracted content.
        """
        pass


# =========================================================
# DOCUMENT CHUNKER INTERFACE
# =========================================================

class DocumentChunkerInterface(ABC):
    """
    Interface for document chunking.

    Responsible for:
    - Splitting parsed documents into chunks
    - Attaching metadata to each chunk
    """

    @abstractmethod
    def chunk(
        self,
        parsed_document: Dict[str, Any],
        logical_filename: str
    ) -> List[Dict[str, Any]]:
        """
        Split a parsed document into chunks suitable for embedding.
        """
        pass


# =========================================================
# TEMPORARY STORAGE INTERFACE
# =========================================================

class LocalTempStorageInterface(ABC):
    """
    Interface for temporary file storage.

    Responsible for:
    - Saving temporary files during processing
    - Cleaning up temporary files after processing
    """

    @abstractmethod
    def save(self, file_bytes: bytes, suffix: str) -> str:
        """
        Save a temporary file and return its path.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete a temporary file.
        """
        pass


# =========================================================
# EMBEDDER INTERFACE
# =========================================================

class EmbedderInterface(ABC):
    """
    Interface for embedding generation.

    Responsible for:
    - Converting text into vector embeddings
    """

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """
        pass


# =========================================================
# VECTOR STORE INTERFACE
# =========================================================

class VectorStoreInterface(ABC):
    """
    Interface for vector database operations.

    Responsible for:
    - Storing embeddings
    - Retrieving relevant vectors
    - Managing indexed files
    """

    # ==============================
    # BASIC OPERATIONS
    # ==============================

    @abstractmethod
    def upsert(self, ids, documents, metadatas, embeddings):
        """
        Insert or update embeddings in the vector store.
        """
        pass

    @abstractmethod
    def query(self, query_embeddings, n_results: int):
        """
        Retrieve the most similar documents for a query embedding.
        """
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[str]):
        """
        Delete embeddings by their IDs.
        """
        pass

    @abstractmethod
    def count(self):
        """
        Return the number of stored embeddings.
        """
        pass

    # ==============================
    # GENERIC SEARCH
    # ==============================

    @abstractmethod
    def find_ids(self, where_filter: Dict[str, Any]) -> List[str]:
        """
        Find embedding IDs that match a metadata filter.
        """
        pass

    # ==============================
    # FILE REGISTRY
    # ==============================

    @abstractmethod
    def register_file(self, user_id: str, file_name: str, embeddings_count: int):
        """
        Register a file that has been indexed.
        """
        pass

    @abstractmethod
    def file_exists(self, file_name: str) -> bool:
        """
        Check whether a file has already been indexed.
        """
        pass

    @abstractmethod
    def is_owner(self, user_id: str, file_name: str) -> bool:
        """
        Verify if a user is the owner of a file.
        """
        pass

    @abstractmethod
    def list_files(self) -> List[Dict[str, Any]]:
        """
        Return the list of indexed files.
        """
        pass

    @abstractmethod
    def remove_file_record_by_id(self, user_id: str, file_name: str):
        """
        Remove a file record for a specific user.
        """
        pass

    @abstractmethod
    def remove_file_record(self, file_name: str):
        """
        Remove all records associated with a file.
        """
        pass


# =========================================================
# QUERY GENERATOR INTERFACE
# =========================================================

class QueryGeneratorInterface(ABC):
    """
    Interface for query generation.

    Responsible for:
    - Expanding user questions into multiple search queries
    """

    @abstractmethod
    def generate(self, question: str) -> List[str]:
        """
        Generate multiple search queries from a user question.
        """
        pass


# =========================================================
# RESPONSE GENERATOR INTERFACE
# =========================================================

class ResponseGeneratorInterface(ABC):
    """
    Interface for RAG response generation.

    Responsible for:
    - Generating partial answers from retrieved context
    - Synthesizing a final response
    """

    @abstractmethod
    def generate(self, question: str, contexts: List[str]) -> List[str]:
        """
        Generate partial answers from retrieved contexts.
        """
        pass




