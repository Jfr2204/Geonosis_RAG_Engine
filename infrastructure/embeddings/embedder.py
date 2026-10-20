import torch
import numpy as np
from pathlib import Path
from RAG_Engine.contracts.interfaces import EmbedderInterface

class Embedder(EmbedderInterface):
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        BASE_DIR = Path(__file__).resolve().parents[2]
        comp_path = BASE_DIR / "resources" / "PCA_components.npy"
        mean_path = BASE_DIR / "resources" / "PCA_mean.npy"

        self.pca_enabled = False
        if comp_path.exists() and mean_path.exists():
            self.pca_components = torch.from_numpy(np.load(comp_path)).to(self.device).float()
            self.pca_mean = torch.from_numpy(np.load(mean_path)).to(self.device).float()
            self.pca_enabled = True
            print(f"✨ PCA cargado en {self.device}")

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        # 1. Obtener la lista de la API
        raw_list = self.embedding_function(texts)
        
        # 2. Convertir primero a un único array de NumPy para eliminar el warning
        raw_np = np.asarray(raw_list, dtype=np.float32)
        
        # 3. Convertir a Tensor de PyTorch y mover a GPU
        raw_tensor = torch.from_numpy(raw_np).to(self.device)

        if not self.pca_enabled:
            return raw_tensor.cpu().numpy()

        # 4. Operaciones en GPU
        centered = raw_tensor - self.pca_mean
        reduced = torch.matmul(centered, self.pca_components.T)
        
        # 5. Volver a NumPy para el resto del pipeline (ChromaDB espera NumPy/listas)
        return reduced.cpu().numpy()
