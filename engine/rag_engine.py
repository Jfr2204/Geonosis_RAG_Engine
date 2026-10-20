from RAG_Engine.contracts.interfaces import (
    RAGEngineInterface,
    DocumentProcessingInterface, 
    QuestionProcessingInterface
)

class RAGEngine(RAGEngineInterface):

    def __init__(
        self,
        document_indexer: DocumentProcessingInterface,
        question_answerer: QuestionProcessingInterface,
    ):
        self.document_indexer = document_indexer
        self.question_answerer = question_answerer

    # ==============================
    # INDEXACIÓN
    # ==============================

    def load_document(self, file_bytes: bytes, filename: str, user_id: str):
        return self.document_indexer.index(
            file_bytes,
            filename,
            user_id
        )

    # ==============================
    # CONSULTA
    # ==============================

    def ask(self, question: str, conversation_context=None, n_results=3):
        return self.question_answerer.answer(
            question,
            conversation_context,
            n_results
        )

    # ==============================
    # ELIMINACIÓN (USUARIO)
    # ==============================

    def delete_document(self, filename: str, user_id: str):
        return self.document_indexer.delete(
            filename,
            user_id
        )

    # ==============================
    # ELIMINACIÓN ADMIN
    # ==============================

    def delete_document_as_admin(self, filename: str):
        return self.document_indexer.delete_as_admin(
            filename
        )

    # ==============================
    # LISTADO
    # ==============================

    def list_documents(self):
        return self.document_indexer.list_documents()
