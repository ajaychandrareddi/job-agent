"""Validation of uploaded resume files (type sniffing, size and zip-bomb limits)."""

from __future__ import annotations

import io
import re
import zipfile

from job_agent.models.resume import ResumeFileType

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
MAX_DOCX_UNCOMPRESSED = 50 * 1024 * 1024
MAX_DOCX_ENTRIES = 2000
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


class UploadRejected(ValueError):
    """Raised when an uploaded file fails validation. Message is safe to show users."""


def sanitize_filename(filename: str) -> str:
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _SAFE_NAME.sub("_", base).strip(" .")
    return (cleaned or "resume")[:200]


def detect_file_type(filename: str, content: bytes, *, max_bytes: int) -> ResumeFileType:
    """Validate an upload and return its type. Content, not extension, is authoritative."""
    if not content:
        raise UploadRejected("file is empty")
    if len(content) > max_bytes:
        raise UploadRejected(f"file exceeds the {max_bytes} byte limit")

    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if content.startswith(PDF_MAGIC):
        if ext not in ("pdf", ""):
            raise UploadRejected("file extension does not match PDF content")
        return ResumeFileType.PDF

    if content.startswith(ZIP_MAGIC):
        if ext not in ("docx", ""):
            raise UploadRejected("file extension does not match DOCX content")
        _check_docx_container(content)
        return ResumeFileType.DOCX

    raise UploadRejected("unsupported file type; upload a PDF or DOCX resume")


def _check_docx_container(content: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_DOCX_ENTRIES:
                raise UploadRejected("DOCX archive has too many entries")
            if sum(i.file_size for i in infos) > MAX_DOCX_UNCOMPRESSED:
                raise UploadRejected("DOCX archive expands beyond the allowed size")
            names = {i.filename for i in infos}
    except zipfile.BadZipFile as exc:
        raise UploadRejected("DOCX file is corrupt") from exc
    if "word/document.xml" not in names:
        raise UploadRejected("ZIP file is not a Word document")
    if any(n.lower().endswith("vbaproject.bin") for n in names):
        raise UploadRejected("macro-enabled documents are not accepted")
