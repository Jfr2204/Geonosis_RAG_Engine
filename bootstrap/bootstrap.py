from RAG_Engine.engine.rag_engine import RAGEngine

from RAG_Engine.services.document_processing import DocumentProcessing
from RAG_Engine.services.question_processing import QuestionProcessing

from RAG_Engine.infrastructure.ingestion.document_parser import DocumentParser
from RAG_Engine.infrastructure.ingestion.document_chunker import DocumentChunker
from RAG_Engine.infrastructure.embeddings.embedder import Embedder
from RAG_Engine.infrastructure.storage.vector_store import ChromaVectorStore
from RAG_Engine.infrastructure.storage.local_temp_storage import LocalTempStorage
from RAG_Engine.infrastructure.storage.storage_quota import StorageQuotaManager  
from RAG_Engine.infrastructure.response.query_generator import QueryGenerator
from RAG_Engine.infrastructure.response.response_generator import ResponseGenerator

from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from pathlib import Path
from dotenv import load_dotenv


def build_embedder(api_key: str, model_name: str) -> Embedder:
    """Crea el embedder usando el modelo específico de OpenAI."""
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=model_name
    )

    return Embedder(
        embedding_function=embedding_function
    )

def build_rag_engine() -> RAGEngine:
    # Project root
    BASE_DIR = Path(__file__).resolve().parents[2]

    # Load .env
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")

    # ==============================
    # Optimization settings
    # ==============================
    EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_DIM   = int(os.getenv("RAG_EMBEDDING_DIM", "512")) 
    BATCH_SIZE      = int(os.getenv("RAG_BATCH_SIZE", "512"))
    CHUNK_SIZE      = int(os.getenv("RAG_CHUNK_SIZE", "450"))

    # Vector DB path
    db_path = os.getenv("CHROMA_DB_PATH") 

    # ==============================
    # Shared infrastructure
    # ==============================

    vector_store = ChromaVectorStore(db_path)

    embedder = build_embedder(api_key, EMBEDDING_MODEL)

    # ==============================
    # Document processing
    # ==============================

    temp_storage = LocalTempStorage()

    quota_manager = StorageQuotaManager(
        vector_store=vector_store,
        db_path=str(db_path)
    )

    parser = DocumentParser()

    chunker = DocumentChunker(
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=CHUNK_SIZE
    )

    # Inyectamos EMBEDDING_DIM y BATCH_SIZE directamente al servicio
    document_processing = DocumentProcessing(
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        temp_storage=temp_storage,
        quota_manager=quota_manager,
        embedding_dim=EMBEDDING_DIM,
        batch_size=BATCH_SIZE
    )

    # ==============================
    # Question processing
    # ==============================

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )

    query_prompt = ChatPromptTemplate.from_template(
        (
            "You are a domain-specific assistant specialized exclusively in:\n"
            "- diseases\n"
            "- genetic disorders\n"
            "- genes and gene–phenotype relationships\n"
            "- molecular and clinical genetics\n\n"

            "TASK:\n"
            "Given a user question, generate search queries ONLY if the question "
            "is related to the allowed domain.\n\n"

            "RULES:\n"
            "1. If the question is NOT related to diseases or genes, output NOTHING.\n"
            "2. If it IS related, generate exactly {n_queries} search queries.\n"
            "3. Each query must be on its own line.\n"
            "4. Do not add explanations, numbering, or extra text.\n\n"

            "User question:\n"
            "{question}"
        )
    )

    response_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
            "You are a domain-specific assistant specialized exclusively in:\n"
            "- human diseases\n"
            "- genetic disorders\n"
            "- genes and gene–phenotype relationships\n"
            "- molecular and clinical genetics\n\n"

            "IMPORTANT LANGUAGE RULE:\n"
            "You MUST always respond in the SAME LANGUAGE as the user's question.\n\n"

            "PRIMARY KNOWLEDGE SOURCE:\n"
            "- The provided context is your MAIN and PRIORITY source of information.\n"
            "- You should base your answer primarily on this context.\n\n"

            "EXTERNAL KNOWLEDGE POLICY:\n"
            "- You MAY use external knowledge ONLY if the context is clearly insufficient.\n"
            "- If you do so, you MUST explicitly indicate it using phrases like:\n"
            "  'Based on general knowledge' or 'Using external knowledge'.\n"
            "- Never mix external knowledge silently with context-based information.\n\n"

            "TASK:\n"
            "Answer the user question using the provided context.\n\n"

            "STYLE AND TONE:\n"
            "- Write in a natural, friendly, and conversational way.\n"
            "- Aim for clarity and helpfulness rather than technical complexity.\n"
            "- Use short paragraphs and smooth explanations.\n"
            "- When helpful, briefly explain concepts instead of only listing facts.\n"
            "- The answer should feel like a knowledgeable human explaining the topic.\n\n"

            "CITATION RULES:\n"
            "- Every statement MUST be supported by the context.\n"
            "- Each piece of context includes a source in the format: [SOURCE: file_name].\n"
            "- You MUST cite sources using that file name.\n"
            "- Use a clear format like: (Source: file_name).\n"
            "- If multiple sources support a statement, include multiple citations.\n"
            "- Do NOT invent or modify source names.\n\n"

            "ANSWER STRUCTURE:\n"
            "- Start with a direct and clear answer.\n"
            "- Then expand with explanation if needed.\n"
            "- For list-type questions, use bullet points.\n"
            "- Keep the answer organized and easy to read.\n\n"

            "RULES:\n"
            "1. If the question is NOT related to diseases, genes, or genetics, politely explain that "
            "you can only answer questions related to diseases and genes.\n"
            "2. Prioritize the provided context as the main source of truth.\n"
            "3. If the context is insufficient, clearly say so before optionally using external knowledge.\n"
            "4. Never fabricate sources or citations.\n"
            "5. Be clear, helpful, and structured.\n\n"

            "Context:\n"
            "{context}\n\n"

            "Question:\n"
            "{question}\n\n"

            "Answer:"
        )
    )

    query_chain = (query_prompt | llm | StrOutputParser() | (lambda x: x.split("\n")))

    query_generator = QueryGenerator(
        query_chain=query_chain,
        n_queries=4
    )

    rag_chain = (response_prompt | llm | StrOutputParser())

    response_generator = ResponseGenerator(
        rag_chain=rag_chain
    )

    question_processing = QuestionProcessing(
        query_generator=query_generator,
        embedder=embedder,
        vector_store=vector_store,
        response_generator=response_generator
    )

    # ==============================
    # Application Layer
    # ==============================

    engine = RAGEngine(
        document_indexer=document_processing,
        question_answerer=question_processing
    )

    return engine
