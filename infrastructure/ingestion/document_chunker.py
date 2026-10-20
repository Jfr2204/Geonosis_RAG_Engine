import re
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from RAG_Engine.contracts.interfaces import DocumentChunkerInterface


class DocumentChunker(DocumentChunkerInterface):

    def __init__(self, tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_tokens: int = 450):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            model_max_length=10000 
        )
        self.max_tokens = max_tokens

        self.hybrid_chunker_instance = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            merge_peers=True,
        )

    # ==========================
    # CHUNKERS MANAGEMENT
    # ==========================

    def chunk(self, parsed_document: dict, logical_filename: str):

        ext = parsed_document["ext"]
        content = parsed_document["content"]

        if ext in [".pdf", ".docx"]:
            return self._docling_chunker(content, logical_filename)

        if ext in [".csv", ".tsv"]:
            return self._csv_chunker(content, logical_filename)

        if ext == ".txt":
            if content["detected_type"] == "natural":
                return self._nl_txt_chunker(content["content"], logical_filename)
            else:
                return self._tabular_txt_chunker(content["content"], logical_filename)

        return self._nl_txt_chunker(content, logical_filename)

    # ==========================
    # CHUNKERS
    # ==========================

    def _docling_chunker(self, documento, source_path):
        chunks = []

        for i, doc_chunk in enumerate(self.hybrid_chunker_instance.chunk(dl_doc=documento)):
            chunks.append({
                "text": doc_chunk.text,
                "metadata": {
                    "source": source_path,
                    "chunk_index": i
                }
            })

        return chunks

    def _csv_chunker(self, text, source_path):
        lines = text.split("\n")
        chunks = []

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            chunks.append({
                "text": line,
                "metadata": {
                    "source": source_path,
                    "row_index": i
                }
            })

        return chunks

    def _nl_txt_chunker(self, text, source_path):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        )

        chunks = []
        for i, chunk in enumerate(splitter.split_text(text)):
            clean = chunk.strip()
            if not clean:
                continue

            chunks.append({
                "text": clean,
                "metadata": {
                    "source": source_path,
                    "chunk_index": i
                }
            })

        return chunks

    def _tabular_txt_chunker(self, text, source_path):
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return []

        headers = re.split(r"\s{2,}|\t", lines[0])

        chunks = []

        for i, line in enumerate(lines[1:], start=1):
            cols = re.split(r"\s{2,}|\t", line)
            row_text = ", ".join(
                f"{h}: {v}" for h, v in zip(headers, cols)
            )

            chunks.append({
                "text": row_text,
                "metadata": {
                    "source": source_path,
                    "row_index": i
                }
            })

        return chunks