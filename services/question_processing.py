from RAG_Engine.contracts.interfaces import (
    QuestionProcessingInterface,
    QueryGeneratorInterface,
    EmbedderInterface,
    VectorStoreInterface,
    ResponseGeneratorInterface
)

from sentence_transformers import CrossEncoder
import re
import torch

class QuestionProcessing(QuestionProcessingInterface):

    def __init__(
        self,
        query_generator: QueryGeneratorInterface,
        embedder: EmbedderInterface,
        vector_store: VectorStoreInterface,
        response_generator: ResponseGeneratorInterface
    ):
        self.query_generator = query_generator
        self.embedder = embedder
        self.vector_store = vector_store
        self.response_generator = response_generator

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=self.device
        )
        print(f" Cross-Encoder cargado en: {self.device}")

    # =========================================================
    # QUERY TYPE DETECTION
    # =========================================================

    def _is_list_question(self, question: str) -> bool:
        keywords = ["qué", "cuales", "cuáles", "which", "what", "list", "genes"]
        return any(k in question.lower() for k in keywords)

    # =========================================================
    # RRF (Reciprocal Rank Fusion)
    # =========================================================

    def _rrf_fusion(self, results_per_query, k: int = 60) -> list[str]:
        scores = {}

        for docs in results_per_query:
            for rank, doc in enumerate(docs):
                if doc not in scores:
                    scores[doc] = 0
                scores[doc] += 1 / (k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked]

    # =========================================================
    # CROSS ENCODER RE-RANKING
    # =========================================================

    def _rerank_with_cross_encoder(self, query: str, documents: list[str], rerank_k: int):
        if not documents:
            return [], []

        documents = documents[:rerank_k]
        pairs = [(query, doc) for doc in documents]

        scores = self.cross_encoder.predict(
            pairs, 
            batch_size=32, 
            show_progress_bar=False
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        docs_ranked = [doc for doc, _ in ranked]
        scores_ranked = [float(score) for _, score in ranked]

        return docs_ranked, scores_ranked

    # =========================================================
    # HELPERS
    # =========================================================

    def _extract_key(self, doc: str) -> str:
        match = re.search(r"gene_symbol:\s*([^,]+)", doc)
        return match.group(1).strip() if match else doc

    def _dynamic_selection(
        self,
        docs: list[str],
        scores: list[float],
        final_k_max: int,
        min_k: int = 3
    ) -> list[str]:

        if not docs:
            return []

        max_score = scores[0]
        min_score = scores[-1]

        score_range = max_score - min_score
        adaptive_threshold = max(0.05, score_range * 0.5)

        diverse_docs = []
        seen_keys = set()

        for doc, score in zip(docs, scores):
            key = self._extract_key(doc)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            diverse_docs.append((doc, score))
            if len(diverse_docs) >= final_k_max:
                break

        filtered = [
            (doc, score)
            for doc, score in diverse_docs
            if score >= max_score - adaptive_threshold
        ]

        if len(filtered) < min_k:
            filtered = diverse_docs[:min_k]

        return [doc for doc, _ in filtered[:final_k_max]]

    # =========================================================
    # MAIN PIPELINE
    # =========================================================

    def answer(
        self,
        question: str,
        conversation_context: str | None,
        retrieval_k: int = 3,
        rerank_k: int = 20,
        final_k_max: int = 8
    ) -> list[str]:

        print("\n==============================")
        print(" NEW QUERY")
        print("==============================")
        print("Question:", question)

        # ── 0) Prompt enrichment ──────────────────────────────
        enriched_prompt = question

        if conversation_context:
            enriched_prompt = (
                "Previous context of the conversation:\n"
                f"{conversation_context}\n\n"
                "current user question:\n"
                f"{question}"
            )

        # ── 1) Query generation ───────────────────────────────
        queries = self.query_generator.generate(enriched_prompt)

        if not queries:
            response = self.response_generator.generate(question, [""])
            return response if isinstance(response, list) else [response]

        # ── 2) Retrieval ──────────────────────────────────────
        results_per_query = []

        for i, query in enumerate(queries):
            embedding = self.embedder.generate_embeddings([query])

            results = self.vector_store.query(
                query_embeddings=embedding.tolist(),
                n_results=retrieval_k
            )

            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            docs_with_meta = []
            for doc, meta in zip(docs, metadatas):
                source = (
                    meta.get("file_name")
                    or meta.get("source")
                    or "UNKNOWN_SOURCE"
                )
                formatted_doc = f"CONTENT:\n{doc}\nSOURCE: {source}"
                docs_with_meta.append(formatted_doc)

            if docs_with_meta:
                results_per_query.append(docs_with_meta)

        # ── 3) RRF ────────────────────────────────────────────
        fused_chunks = self._rrf_fusion(results_per_query)

        # ── 4) Cross-encoder ──────────────────────────────────
        reranked_docs, scores = self._rerank_with_cross_encoder(
            question,
            fused_chunks,
            rerank_k
        )

        # ── 5) Selection strategy ─────────────────────────────
        if self._is_list_question(question):
            final_chunks = reranked_docs[:final_k_max]
        else:
            final_chunks = self._dynamic_selection(reranked_docs, scores, final_k_max)

        # ── 6) Context assembly ───────────────────────────────
        assembled_context = "\n\n".join(
            f"[CONTEXT {i+1}]\n{doc}"
            for i, doc in enumerate(final_chunks)
        )

        final_context = assembled_context
        if conversation_context:
            final_context = (
                "Previous context of the conversation:\n"
                f"{conversation_context}\n\n"
                "retrieved information:\n"
                f"{assembled_context}"
            )

        # ── 7) Final answer ───────────────────────────────────
        response = self.response_generator.generate(
            question,
            [final_context]
        )

        # Devuelve una lista de strings
        if isinstance(response, list):
            return response
        return [response]


