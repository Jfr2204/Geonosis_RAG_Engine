# 🧬 Geonosis RAG Engine

**Motor RAG (Retrieval-Augmented Generation) para la consulta en lenguaje natural de enfermedades raras, genes y fenotipos.**

Este repositorio contiene el **backend de dominio** de *Geonosis*, un sistema inteligente de información genómica y fenotípica para el soporte del diagnóstico de enfermedades raras. El motor se encarga de ingerir documentos, generar y almacenar embeddings, recuperar contexto relevante y generar respuestas fundamentadas con un LLM.

> Proyecto Integrador de Ingeniería en Informática (Universidad Nacional de San Luis) y desarrollo realizado en el marco de las Prácticas Profesionales Supervisadas, en el LIDIC (Laboratorio de Investigación y Desarrollo en Inteligencia Computacional).

---

## 📌 Contexto

Existen unas 7.000 enfermedades raras, más del 80 % de origen genético. Fuentes como Orphanet, OMIM y HPO contienen información valiosa, pero está dispersa y exige búsquedas manuales y conocimiento técnico avanzado. Este motor permite centralizar esa información en una base de conocimiento propia y consultarla en lenguaje natural.

## ✨ Características

- 📂 **Ingesta multiformato**: `.pdf`, `.docx`, `.txt`, `.csv`, `.tsv` (además de `.xml` y `.json` a nivel de parser).
- ✂️ **Chunking adaptado al tipo de contenido**:
  - PDF/DOCX → `HybridChunker` de Docling (tokenizador `all-MiniLM-L6-v2`, 450 tokens máx.).
  - TXT en lenguaje natural → `RecursiveCharacterTextSplitter` (800 caracteres, 100 de solapamiento).
  - CSV/TSV y TXT tabulares → **una fila por chunk**, con formato `columna: valor`.
  - Detección automática de si un `.txt` es natural o tabular.
- 🔢 **Embeddings** con `text-embedding-3-large` de OpenAI y **reducción de dimensionalidad con PCA** (opcional).
- 🗄️ **Base vectorial local** con ChromaDB (HNSW, distancia coseno).
- 🔎 **RAG Fusion**: expansión de la pregunta en múltiples consultas, fusión de rankings con **Reciprocal Rank Fusion (RRF)**, re-ranking con **cross-encoder** y selección dinámica de fragmentos.
- 💬 **Contexto conversacional**: acepta el historial reciente para responder preguntas de seguimiento.
- 🛡️ **Dominio acotado**: los prompts restringen el asistente a enfermedades, genes y genética, y obligan a responder en el idioma de la pregunta y solo con el contexto recuperado.
- 👥 **Gestión de documentos con propiedad**: cada archivo pertenece a un usuario; se puede eliminar como dueño o como administrador.
- 💾 **Cuota de almacenamiento**: estimación y control del espacio antes de indexar.

## 🏗️ Arquitectura

El motor sigue una arquitectura en capas, desacoplada mediante interfaces abstractas (`contracts/interfaces.py`) y con inyección de dependencias en `bootstrap/bootstrap.py`. Esto permite reemplazar el modelo de embeddings, la base vectorial o el LLM sin tocar la lógica del sistema.

```
RAG_Engine/
├── bootstrap/
│   └── bootstrap.py              # build_rag_engine(): ensambla todas las dependencias
├── contracts/
│   └── interfaces.py             # Interfaces abstractas de cada componente
├── engine/
│   └── rag_engine.py             # RAGEngine: fachada pública del motor
├── services/
│   ├── document_processing.py    # Carga, borrado y listado de documentos
│   └── question_processing.py    # Pipeline de respuesta (RAG Fusion)
└── infrastructure/
    ├── ingestion/
    │   ├── document_parser.py    # Parseo por formato
    │   └── document_chunker.py   # Estrategias de chunking
    ├── embeddings/
    │   └── embedder.py           # Embeddings OpenAI + PCA
    ├── storage/
    │   ├── vector_store.py       # ChromaVectorStore (embeddings + registro de archivos)
    │   ├── local_temp_storage.py # Archivos temporales durante la ingesta
    │   ├── storage_estimator.py  # Estimación de espacio de embeddings
    │   └── storage_quota.py      # Control de cuota de almacenamiento
    └── response/
        ├── query_generator.py    # Expansión de consultas
        └── response_generator.py # Respuestas parciales y síntesis final
```

### Flujo de ingesta

```
archivo (bytes) → archivo temporal → parseo → chunking → control de cuota
   → embeddings (por lotes de 512) → upsert en ChromaDB → registro del archivo
```

### Flujo de consulta

1. **Expansión**: un LLM genera *N* consultas alternativas a partir de la pregunta (y devuelve vacío si no pertenece al dominio).
2. **Recuperación**: búsqueda semántica en ChromaDB para cada consulta.
3. **Fusión**: combinación de rankings con RRF.
4. **Re-ranking**: cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`.
5. **Selección dinámica**: umbral adaptativo y diversidad; para preguntas de tipo listado se prioriza la cobertura.
6. **Generación**: el LLM responde condicionado por el contexto y se sintetizan las respuestas parciales en una sola.

## 🛠️ Stack

**Python 3.11** · LangChain · Docling · ChromaDB · OpenAI (`text-embedding-3-large`, `gpt-4o-mini`) · Sentence Transformers / Transformers · NumPy · Pandas · python-dotenv

## 🚀 Instalación y configuración

### Requisitos

- Python 3.11
- Una API key de OpenAI
- Espacio en disco para la base vectorial

### 1. Clonar e instalar dependencias

```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install chromadb langchain-openai langchain-core langchain-text-splitters \
            docling transformers sentence-transformers pandas numpy python-dotenv
```

> Ajustá la lista a tu `pyproject.toml` o `requirements.txt` si ya los tenés definidos.

### 2. Variables de entorno

Creá un archivo `.env` en la **raíz del proyecto** (el directorio que contiene la carpeta `RAG_Engine/`). No lo subas al repositorio.

```env
# Obligatoria
OPENAI_API_KEY=sk-...

# Opcional: tope de almacenamiento de la base vectorial.
# Formato legible (B, KB, MB, GB, TB) o en bytes. Si no se define, no hay límite.
RAG_STORAGE_CAP=50GB
# RAG_STORAGE_CAP_BYTES=53687091200

# Opcional: dimensión de embeddings de respaldo (por defecto 3072)
RAG_EMBEDDING_DIM=3072
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | Sí | Credencial para embeddings y LLM. |
| `RAG_STORAGE_CAP` | No | Límite de almacenamiento en formato legible (ej. `50GB`). Tiene prioridad sobre la siguiente. |
| `RAG_STORAGE_CAP_BYTES` | No | Límite de almacenamiento en bytes. |
| `RAG_EMBEDDING_DIM` | No | Dimensión usada si no se puede calcular automáticamente (por defecto `3072`). |

### 3. Archivos y carpetas esperados

| Ruta | Descripción |
|---|---|
| `<raíz>/.env` | Variables de entorno. |
| `<raíz>/vectordb/` | Base vectorial persistente. Se crea automáticamente. |
| `RAG_Engine/resources/PCA_components.npy` y `PCA_mean.npy` | Matrices de PCA. **Opcionales**: si no existen, se usan los embeddings originales sin reducir. |

> ⚠️ **Importante sobre PCA:** si indexás con PCA activado, las consultas también deben usarlo (y viceversa). Mezclar embeddings reducidos y sin reducir en la misma colección produce resultados incorrectos. Elegí una configuración antes de indexar y mantenela.

### 4. Parámetros configurables en `bootstrap.py`

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| Modelo de embeddings | `text-embedding-3-large` | Cambiable en `build_embedder`. |
| Modelo LLM | `gpt-4o-mini`, `temperature=0` | Cambiable en `build_rag_engine`. |
| Tokenizador del chunker | `sentence-transformers/all-MiniLM-L6-v2` | Usado por `HybridChunker`. |
| `max_tokens` por chunk | `450` | Tamaño máximo para PDF/DOCX. |
| `n_queries` | `4` | Consultas generadas por pregunta. |
| `n_results` | `3` | Fragmentos por consulta (en el sistema completo: Mini=3, Mid=6, Pro=9). |
| Colección ChromaDB | `test_collection_pca` | Nombre de la colección de embeddings. Cambialo para producción. |

## 📖 Uso

```python
from RAG_Engine.bootstrap.bootstrap import build_rag_engine

# Construye el motor con todas sus dependencias.
# Es costoso: crealo una sola vez (por ejemplo, como singleton).
engine = build_rag_engine()

# 1) Indexar un documento
with open("disorder_genes.csv", "rb") as f:
    result = engine.load_document(f.read(), "disorder_genes.csv", "user-1")
print(result)
# {'file_name': 'disorder_genes.csv', 'embeddings_created': 8300,
#  'status': 'success', 'storage_estimate': '...'}

# 2) Consultar
answers = engine.ask(
    "¿Qué genes están asociados al síndrome de Marfan?",
    conversation_context=None,   # opcional: historial reciente en texto
    n_results=3,
)
print(answers[0])

# 3) Listar documentos indexados
print(engine.list_documents())

# 4) Eliminar un documento
engine.delete_document("disorder_genes.csv", "user-1")        # como dueño
engine.delete_document_as_admin("disorder_genes.csv")          # como administrador
```

### Contexto conversacional

`conversation_context` es un texto con los últimos mensajes de la conversación, por ejemplo:

```
User: ¿Qué es el síndrome de Marfan?
Assistant: Es un trastorno genético del tejido conectivo...
```

El motor lo usa tanto para generar las consultas de búsqueda como para responder.

### Manejo de errores

Los métodos lanzan `ValueError` con mensajes descriptivos en casos como: archivo duplicado, fallo de parseo o chunking, cuota de almacenamiento excedida, archivo inexistente o falta de permisos al eliminar. Capturalos en tu capa de aplicación.

## 📂 Formatos soportados y chunking

| Formato | Parser | Estrategia de chunking |
|---|---|---|
| `.pdf` | Docling (sin OCR por defecto) | `HybridChunker` |
| `.docx` | Docling | `HybridChunker` |
| `.csv` / `.tsv` | Pandas | Una fila por chunk (`col: valor, ...`) |
| `.txt` | Texto plano + detección de tipo | Natural: `RecursiveCharacterTextSplitter`. Tabular: una fila por chunk |
| `.xml` / `.json` | Texto estructurado | `RecursiveCharacterTextSplitter` |

## ⚠️ Consideraciones

- **Nombres de archivo únicos**: son el identificador del documento. Subir dos archivos con el mismo nombre es rechazado.
- **Procesos largos**: la indexación de archivos grandes (cientos de miles de filas) genera muchos embeddings y llamadas a la API de OpenAI; tiene costo y puede tardar.
- **Respuestas acotadas al contexto**: si la base no contiene la información, el motor lo indica en lugar de inventarla.
- **No es un dispositivo de diagnóstico**: es una herramienta de apoyo a la investigación. Las respuestas deben ser validadas por profesionales.

## 🗺️ Trabajos futuros

- Búsqueda híbrida (semántica + léxica con BM25).
- Ajuste dinámico del chunking según estructura del documento.
- Archivos temporales por sesión sin persistencia en la base.
- Trazabilidad de respuestas (citar fragmentos y fuentes).
- Procesamiento asíncrono con colas de tareas para escalar.

## 👥 Autores

- **Milko J. Gomez Boquete**
- **Juan Facundo Rivero**

**Director:** Dra. Leticia C. Cagnina · **Co-Directora:** Dra. Lucía Spangenberg
Universidad Nacional de San Luis · Ingeniería en Informática

## 📄 Licencia

Software propietario/privado.
