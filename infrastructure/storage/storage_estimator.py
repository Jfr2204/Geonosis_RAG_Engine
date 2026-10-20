from dataclasses import dataclass
 
# ── Tuneable constants ──────────────────────────────────────────────────────
HNSW_OVERHEAD_FACTOR: float = 0.25   # HNSW graph adds ~25 % over raw vectors
AVG_METADATA_BYTES:   int   = 512    # avg bytes per chunk for metadata/doc text
 
 
@dataclass(frozen=True)
class StorageEstimate:
    n_chunks:           int
    embedding_dim:      int
    raw_vectors_bytes:  int
    overhead_bytes:     int
    metadata_bytes:     int
    total_bytes:        int
 
    # ── Human-readable helpers ──────────────────────────────────────────────
 
    @staticmethod
    def _fmt(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.2f} {unit}"
            n /= 1024
        return f"{n:.2f} TB"
 
    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)
 
    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)
 
    def summary(self) -> str:
        return (
            f"Chunks: {self.n_chunks:,} | "
            f"Vectors: {self._fmt(self.raw_vectors_bytes)} | "
            f"HNSW overhead: {self._fmt(self.overhead_bytes)} | "
            f"Metadata: {self._fmt(self.metadata_bytes)} | "
            f"Total estimate: {self._fmt(self.total_bytes)}"
        )
 
 
def estimate_storage(
    n_chunks: int,
    embedding_dim: int,
    hnsw_factor: float = HNSW_OVERHEAD_FACTOR,
    avg_metadata_bytes: int = AVG_METADATA_BYTES,
) -> StorageEstimate:
    """Return a StorageEstimate for the given chunk/dim combination."""
    raw   = n_chunks * embedding_dim * 4          # float32 = 4 bytes
    ohd   = int(raw * hnsw_factor)
    meta  = n_chunks * avg_metadata_bytes
    total = raw + ohd + meta
 
    return StorageEstimate(
        n_chunks=n_chunks,
        embedding_dim=embedding_dim,
        raw_vectors_bytes=raw,
        overhead_bytes=ohd,
        metadata_bytes=meta,
        total_bytes=total,
    )