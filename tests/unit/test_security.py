from __future__ import annotations

import io
import json
import zipfile

import pytest

from job_agent.models.resume import ResumeFileType
from job_agent.security.redaction import redact, redact_text
from job_agent.security.uploads import UploadRejected, detect_file_type, sanitize_filename

MAX = 5 * 1024 * 1024


class TestUploads:
    def test_pdf_and_docx_detected(self, sample_pdf: bytes, sample_docx: bytes) -> None:
        assert detect_file_type("cv.pdf", sample_pdf, max_bytes=MAX) is ResumeFileType.PDF
        assert detect_file_type("cv.docx", sample_docx, max_bytes=MAX) is ResumeFileType.DOCX

    def test_extension_mismatch_rejected(self, sample_pdf: bytes) -> None:
        with pytest.raises(UploadRejected, match="does not match"):
            detect_file_type("cv.docx", sample_pdf, max_bytes=MAX)

    @pytest.mark.parametrize("content", [b"", b"hello world", b"\x89PNG\r\n"])
    def test_bad_content_rejected(self, content: bytes) -> None:
        with pytest.raises(UploadRejected):
            detect_file_type("cv.pdf", content, max_bytes=MAX)

    def test_size_limit(self, sample_pdf: bytes) -> None:
        with pytest.raises(UploadRejected, match="limit"):
            detect_file_type("cv.pdf", sample_pdf, max_bytes=len(sample_pdf) - 1)

    def test_non_word_zip_rejected(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a.txt", "hi")
        with pytest.raises(UploadRejected, match="not a Word document"):
            detect_file_type("cv.docx", buf.getvalue(), max_bytes=MAX)

    def test_macro_docx_rejected(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("word/document.xml", "<x/>")
            zf.writestr("word/vbaProject.bin", "x")
        with pytest.raises(UploadRejected, match="macro"):
            detect_file_type("cv.docx", buf.getvalue(), max_bytes=MAX)

    def test_zip_bomb_rejected(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("word/document.xml", "<x/>")
            zf.writestr("word/big.bin", b"\0" * (51 * 1024 * 1024))
        with pytest.raises(UploadRejected, match="expands"):
            detect_file_type("cv.docx", buf.getvalue(), max_bytes=MAX)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../etc/passwd", "passwd"),
            ("C:\\x\\my cv.pdf", "my cv.pdf"),
            ("<script>.pdf", "_script_.pdf"),
            ("...", "resume"),
        ],
    )
    def test_sanitize_filename(self, raw: str, expected: str) -> None:
        assert sanitize_filename(raw) == expected


class TestRedaction:
    def test_text_patterns(self) -> None:
        out = redact_text("mail jane@x.io call (555) 123-4567 ssn 123-45-6789 Bearer abcdefghijk")
        assert "jane@x.io" not in out and "123-4567" not in out and "6789" not in out
        assert "abcdefghijk" not in out

    def test_sensitive_keys_dropped(self) -> None:
        payload = {
            "resume_text": "secret",
            "nested": {"answer": "yes", "count": 3},
            "items": ["a@b.co"],
        }
        out = redact(payload)
        assert out == {
            "resume_text": "[REDACTED]",
            "nested": {"answer": "[REDACTED]", "count": 3},
            "items": ["[EMAIL]"],
        }

    def test_logging_pipeline_redacts(self, capsys: pytest.CaptureFixture[str]) -> None:
        import structlog

        from job_agent.observability.logging import configure_logging, get_logger

        structlog.reset_defaults()
        configure_logging("INFO", json=True)
        get_logger("t").info("uploaded jane@x.io", extracted_text="full resume", n=1)
        line = capsys.readouterr().out.strip().splitlines()[-1]
        record = json.loads(line)
        assert record["extracted_text"] == "[REDACTED]"
        assert record["event"] == "uploaded [EMAIL]"
        assert record["n"] == 1
        structlog.reset_defaults()
