import tempfile
from pathlib import Path
from RAG_Engine.contracts.interfaces import LocalTempStorageInterface


class LocalTempStorage(LocalTempStorageInterface):

    def save(self, file_bytes: bytes, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            return tmp.name

    def delete(self, path: str) -> None:
        Path(path).unlink(missing_ok=True)