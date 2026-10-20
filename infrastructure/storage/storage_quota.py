import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from RAG_Engine.infrastructure.storage.storage_estimator import (
    StorageEstimate,
    estimate_storage,
)


# =========================================================
# Helpers
# =========================================================

def _parse_bytes(raw: str) -> int:
    raw = raw.strip()
    multipliers = {
        "TB": 1024 ** 4,
        "GB": 1024 ** 3,
        "MB": 1024 ** 2,
        "KB": 1024 ** 1,
        "B":  1,
    }

    upper = raw.upper()
    for suffix, factor in multipliers.items():
        if upper.endswith(suffix):
            number = float(raw[: -len(suffix)])
            return int(number * factor)
        
    return int(raw)


def get_directory_size_bytes(path: Path) -> int:
    total_size = 0
    path = Path(path)
    
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_size += os.path.getsize(fp)
            except (OSError, FileNotFoundError):
                continue
                
    return total_size


# =========================================================
# Data model
# =========================================================

@dataclass(frozen=True)
class QuotaCheckResult:
    allowed:          bool
    estimate:         StorageEstimate
    used_bytes:       int
    available_bytes:  Optional[int]
    cap_bytes:        Optional[int]
    warning_message:  Optional[str]


# =========================================================
# Main class
# =========================================================

class StorageQuotaManager:

    ENV_HUMAN = "RAG_STORAGE_CAP"
    ENV_BYTES = "RAG_STORAGE_CAP_BYTES"

    def __init__(
        self,
        vector_store=None,
        db_path: str | None = None,
    ):
        self._vector_store = vector_store
        self._cap_bytes: Optional[int] = self._read_cap()
        self._db_path = Path(db_path) if db_path else None

    # =========================================================
    # PUBLIC
    # =========================================================

    @property
    def cap_bytes(self) -> Optional[int]:
        return self._cap_bytes

    def get_used_bytes(self, embedding_dim: int) -> int:
        """
        Uses hybrid strategy:
        max(real_disk_usage, estimated_usage)
        """

        real_size = 0
        estimated_size = 0

        # ── Real disk usage ───────────────────────────────
        if self._db_path and self._db_path.exists():
            try:
                real_size = get_directory_size_bytes(self._db_path)
            except Exception:
                real_size = 0

        # ── Estimated usage ───────────────────────────────
        if self._vector_store:
            files = self._vector_store.list_files()
            for f in files:
                n = f.get("embeddings_count", 0)
                est = estimate_storage(n_chunks=n, embedding_dim=embedding_dim)
                estimated_size += est.total_bytes

        # ── Hybrid decision ───────────────────────────────
        return max(real_size, estimated_size)

    def check(
        self,
        n_chunks: int,
        embedding_dim: int,
    ) -> QuotaCheckResult:

        estimate   = estimate_storage(n_chunks, embedding_dim)
        used_bytes = self.get_used_bytes(embedding_dim)

        cap       = self._cap_bytes
        available = (cap - used_bytes) if cap is not None else None

        if cap is None:
            return QuotaCheckResult(
                allowed=True,
                estimate=estimate,
                used_bytes=used_bytes,
                available_bytes=None,
                cap_bytes=None,
                warning_message=None,
            )

        will_exceed = (used_bytes + estimate.total_bytes) > cap
        warning: Optional[str] = None

        if will_exceed:
            warning = (
                f"- Storage quota exceeded:\n"
                f"- Required: ~{estimate.total_mb:.1f} MB\n"
                f"- Available: {available / (1024**2):.1f} MB\n"
                f"- Used: {used_bytes / (1024**3):.2f} GB\n"
                f"- Cap: {cap / (1024**3):.2f} GB"
            )

        return QuotaCheckResult(
            allowed=not will_exceed,
            estimate=estimate,
            used_bytes=used_bytes,
            available_bytes=available,
            cap_bytes=cap,
            warning_message=warning,
        )

    # =========================================================
    # PRIVATE
    # =========================================================

    @staticmethod
    def _read_cap() -> Optional[int]:
        human = os.getenv(StorageQuotaManager.ENV_HUMAN, "").strip()
        if human:
            return _parse_bytes(human)

        raw = os.getenv(StorageQuotaManager.ENV_BYTES, "").strip()
        if raw:
            return int(raw)

        return None