import json
import pandas as pd
import re
import xml.etree.ElementTree as ET
import torch
from pathlib import Path
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions
)
from RAG_Engine.contracts.interfaces import DocumentParserInterface


class DocumentParser(DocumentParserInterface):

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if self.cuda_available else "CPU"
        print(f" Inicializando DocumentParser. Dispositivo detectado: {device_name}")

        self.pdf_pipeline = self._crear_pipeline_pdf()

        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF, 
                InputFormat.DOCX
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pdf_pipeline
                ),
                InputFormat.DOCX: WordFormatOption()
            }
        )

        self.parsers = {
            ".pdf": self._pdf_processing,
            ".docx": self._docx_processing,
            ".csv": self._csv_tsv_processing,
            ".tsv": self._csv_tsv_processing,
            ".txt": self._txt_processing,
            ".xml": self._xml_processing,
            ".json": self._json_processing
        }

    def _crear_pipeline_pdf(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        options = PdfPipelineOptions(
            generate_page_images=False,
            do_ocr=False,
            do_picture_description=False,
        )
        
        options.accelerator_options.device = device
        
        return options
    
    # ==========================
    # PARSERS MANAGEMENT
    # ==========================

    def parse(self, file_path: str):
        ext = Path(file_path).suffix.lower()

        if ext not in self.parsers:
            raise ValueError(f"Formato no soportado: {ext}")

        return {
            "ext": ext,
            "content": self.parsers[ext](file_path)
        }

    # ==========================
    # PARSERS (Optimized)
    # ==========================

    def _pdf_processing(self, ruta_archivo):
        if self.cuda_available:
            torch.cuda.empty_cache()
            mem_inicial = torch.cuda.memory_allocated(0) / 1024**2

        result = self.converter.convert(Path(ruta_archivo))

        if self.cuda_available:
            mem_final = torch.cuda.memory_allocated(0) / 1024**2
    
        return result.document

    def _docx_processing(self, ruta_archivo):
        result = self.converter.convert(Path(ruta_archivo))
        return result.document

    def _csv_tsv_processing(self, ruta_archivo):
        ext = Path(ruta_archivo).suffix.lower()
        df = pd.read_csv(ruta_archivo, sep='\t' if ext == '.tsv' else ',')
        filas_texto = [
            ", ".join(f"{col}: {fila[col]}" for col in df.columns if pd.notna(fila[col]))
            for _, fila in df.iterrows()
        ]
        return "\n".join(filas_texto)

    def _txt_processing(self, ruta_archivo):
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            text = f.read()

        return {
            "content": text,
            "detected_type": self._detect_content_type(text)
        }

    def _xml_processing(self, ruta_archivo):
        tree = ET.parse(ruta_archivo)
        root = tree.getroot()
        return ET.tostring(root, encoding='unicode')

    def _json_processing(self, ruta_archivo):
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _detect_content_type(self, text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return "natural"

        sep_pattern = r"[,\t;|]"
        tabular_count = sum(1 for l in lines if len(re.findall(sep_pattern, l)) >= 2)

        if tabular_count / len(lines) >= 0.5:
            return "tabular"

        return "natural"
    
