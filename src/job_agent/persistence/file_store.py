"""Storage for original resume files. Local disk in dev; swap for object storage
(S3/GCS with server-side encryption) in deployed environments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol
from uuid import UUID


class ResumeFileStore(Protocol):
    def save(
        self, user_id: UUID, resume_id: UUID, version: int, sha256: str, content: bytes, suffix: str
    ) -> str: ...

    def load(self, ref: str) -> bytes: ...

    def delete(self, ref: str) -> None: ...


class LocalResumeFileStore:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _path(self, ref: str) -> Path:
        path = (self._root / ref).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("storage reference escapes the storage root")
        return path

    def save(
        self, user_id: UUID, resume_id: UUID, version: int, sha256: str, content: bytes, suffix: str
    ) -> str:
        ref = f"{user_id}/{resume_id}/v{version}-{sha256[:16]}.{suffix}"
        path = self._path(ref)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        return ref

    def load(self, ref: str) -> bytes:
        return self._path(ref).read_bytes()

    def delete(self, ref: str) -> None:
        self._path(ref).unlink(missing_ok=True)


class InMemoryResumeFileStore:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def save(
        self, user_id: UUID, resume_id: UUID, version: int, sha256: str, content: bytes, suffix: str
    ) -> str:
        ref = f"{user_id}/{resume_id}/v{version}-{sha256[:16]}.{suffix}"
        self.files[ref] = content
        return ref

    def load(self, ref: str) -> bytes:
        return self.files[ref]

    def delete(self, ref: str) -> None:
        self.files.pop(ref, None)
