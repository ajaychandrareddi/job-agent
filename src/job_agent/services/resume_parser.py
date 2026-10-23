"""Resume parsing pipeline: validate -> extract text -> structured profile -> grounding.

Grounding is the anti-fabrication guard: any skill, title, certification, employer,
degree or accomplishment that cannot be located in the resume text is dropped and
recorded as a warning. Years of experience are recomputed from stated dates.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date

from job_agent.models.common import utcnow
from job_agent.models.resume import (
    Education,
    ParsedResume,
    Position,
    ResumeProfile,
)
from job_agent.observability.logging import get_logger
from job_agent.security.uploads import detect_file_type
from job_agent.services.resume_extractors import (
    ExtractionFailed,
    HeuristicResumeExtractor,
    ResumeExtractor,
)
from job_agent.services.text_extraction import extract_text

log = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9+#.]+")
PHRASE_COVERAGE = 0.85  # share of a sentence's words that must appear in the resume
YEARS_TOLERANCE = 1.0


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class TextGrounder:
    """Checks whether phrases/sentences are supported by a source text."""

    def __init__(self, text: str) -> None:
        self._text = " ".join(text.lower().split())
        self._words = set(_WORD.findall(self._text))
        self.warnings: list[str] = []

    def contains(self, phrase: str) -> bool:
        needle = " ".join(phrase.lower().split())
        if not needle:
            return False
        pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9+#])"
        return re.search(pattern, self._text) is not None

    def covered(self, sentence: str) -> bool:
        words = [w.strip(".") for w in _WORD.findall(sentence.lower())]
        words = [w for w in words if w]
        if not words:
            return False
        hits = sum(1 for w in words if w in self._words or f"{w}." in self._words)
        return hits / len(words) >= PHRASE_COVERAGE

    def keep(self, field: str, items: list[str], *, phrase: bool = False) -> list[str]:
        kept: list[str] = []
        for item in items:
            ok = self.covered(item) if phrase else self.contains(item)
            if ok:
                kept.append(item)
            else:
                self.warnings.append(f"dropped unsupported {field}: {item[:80]!r}")
        return kept

    def opt(self, field: str, value: str | None) -> str | None:
        if value is None or self.contains(value):
            return value
        self.warnings.append(f"dropped unsupported {field}: {value[:80]!r}")
        return None

    def year_present(self, d: date | None) -> bool:
        return d is None or str(d.year) in self._text


def ground_profile(
    profile: ResumeProfile, text: str, *, today: date
) -> tuple[ResumeProfile, list[str]]:
    g = TextGrounder(text)

    positions: list[Position] = []
    for p in profile.positions:
        if not g.contains(p.title):
            g.warnings.append(f"dropped position with unsupported title: {p.title[:80]!r}")
            continue
        start, end = p.start_date, p.end_date
        if not (g.year_present(start) and g.year_present(end)):
            g.warnings.append(f"dropped unverifiable dates for position {p.title[:80]!r}")
            start, end = None, None
        if start and end and end < start:
            g.warnings.append(f"dropped inverted dates for position {p.title[:80]!r}")
            start, end = None, None
        positions.append(
            Position(
                title=p.title,
                company=g.opt("company", p.company),
                start_date=start,
                end_date=end,
                is_current=p.is_current and start is not None,
                highlights=g.keep("highlight", p.highlights, phrase=True),
            )
        )

    education = [
        Education(
            institution=g.opt("institution", e.institution),
            degree=g.opt("degree", e.degree),
            field_of_study=g.opt("field_of_study", e.field_of_study),
            graduation_year=e.graduation_year
            if e.graduation_year is None or str(e.graduation_year) in text
            else None,
        )
        for e in profile.education
    ]
    education = [e for e in education if e.institution or e.degree or e.field_of_study]

    job_titles = g.keep("job_title", profile.job_titles)
    for p in positions:
        if p.title.lower() not in {t.lower() for t in job_titles}:
            job_titles.append(p.title)

    grounded = ResumeProfile(
        full_name=g.opt("full_name", profile.full_name),
        headline=profile.headline if profile.headline and g.covered(profile.headline) else None,
        summary=profile.summary if profile.summary and g.covered(profile.summary) else None,
        skills=g.keep("skill", profile.skills),
        technologies=g.keep("technology", profile.technologies),
        certifications=g.keep("certification", profile.certifications, phrase=True),
        job_titles=job_titles,
        positions=positions,
        education=education,
        accomplishments=g.keep("accomplishment", profile.accomplishments, phrase=True),
        domains=g.keep("domain", profile.domains),
        years_of_experience=None,
    )
    years, years_warning = _reconcile_years(profile.years_of_experience, positions, text, today)
    grounded.years_of_experience = years
    if years_warning:
        g.warnings.append(years_warning)
    return grounded, g.warnings


def compute_experience_years(positions: list[Position], today: date) -> float | None:
    """Total years covered by dated positions, merging overlaps."""
    intervals: list[tuple[date, date]] = []
    for p in positions:
        if p.start_date is None:
            continue
        end = p.end_date or (today if p.is_current else None)
        if end is None or end < p.start_date:
            continue
        intervals.append((p.start_date, min(end, today)))
    if not intervals:
        return None
    intervals.sort()
    merged: list[list[date]] = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    days = sum((end - start).days for start, end in merged)
    return round(days / 365.25, 1)


_STATED_YEARS = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs)", re.I)


def _reconcile_years(
    claimed: float | None, positions: list[Position], text: str, today: date
) -> tuple[float | None, str | None]:
    computed = compute_experience_years(positions, today)
    if computed is not None:
        if claimed is not None and abs(claimed - computed) > YEARS_TOLERANCE:
            return computed, (
                f"years_of_experience {claimed} replaced by {computed} computed from dates"
            )
        return computed, None
    if claimed is None:
        return None, None
    stated = {float(m.group(1)) for m in _STATED_YEARS.finditer(text)}
    if any(abs(claimed - s) <= 0.01 for s in stated):
        return claimed, None
    return None, f"dropped unsupported years_of_experience {claimed}"


class ResumeParser:
    def __init__(
        self,
        extractor: ResumeExtractor | None = None,
        *,
        fallback: ResumeExtractor | None = None,
        max_upload_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        self._extractor = extractor or HeuristicResumeExtractor()
        self._fallback = fallback if fallback is not None else HeuristicResumeExtractor()
        self._max_bytes = max_upload_bytes

    def parse(self, filename: str, content: bytes, *, today: date | None = None) -> ParsedResume:
        file_type = detect_file_type(filename, content, max_bytes=self._max_bytes)
        text = extract_text(content, file_type)
        warnings: list[str] = []
        try:
            profile = self._extractor.extract(text)
        except ExtractionFailed as exc:
            log.warning("resume_extraction_fallback", reason=str(exc))
            warnings.append(f"primary extractor failed ({exc}); used deterministic fallback")
            profile = self._fallback.extract(text)
        grounded, grounding_warnings = ground_profile(profile, text, today=today or utcnow().date())
        warnings.extend(grounding_warnings)
        log.info(
            "resume_parsed",
            file_type=file_type.value,
            skills=len(grounded.skills),
            positions=len(grounded.positions),
            dropped=len(grounding_warnings),
        )
        return ParsedResume(
            profile=grounded,
            text=text,
            content_sha256=sha256_hex(content),
            file_type=file_type,
            warnings=warnings,
        )
