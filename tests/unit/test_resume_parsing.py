from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

from job_agent.models.resume import Education, Position, ResumeFileType, ResumeProfile
from job_agent.services.resume_extractors import (
    ExtractionFailed,
    HeuristicResumeExtractor,
    LLMResumeExtractor,
)
from job_agent.services.resume_parser import (
    ResumeParser,
    compute_experience_years,
    ground_profile,
)
from job_agent.services.text_extraction import (
    TextExtractionError,
    extract_text,
    normalize_text,
)
from tests.conftest import SAMPLE_RESUME_LINES, TODAY, make_docx

SAMPLE_TEXT = "\n".join(SAMPLE_RESUME_LINES)


class StaticExtractor:
    def __init__(self, profile: ResumeProfile) -> None:
        self.profile = profile

    def extract(self, text: str) -> ResumeProfile:
        return self.profile


class FailingExtractor:
    def extract(self, text: str) -> ResumeProfile:
        raise ExtractionFailed("boom")


class FakeStructuredModel:
    """Minimal stand-in for a chat model's ``with_structured_output``."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.prompts: list[Any] = []

    def with_structured_output(self, schema: type) -> RunnableLambda[Any, Any]:
        def run(prompt_value: Any) -> Any:
            self.prompts.append(prompt_value)
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

        return RunnableLambda(run)


# --------------------------------------------------------------------------- text


class TestTextExtraction:
    def test_docx_including_tables(self) -> None:
        content = make_docx(SAMPLE_RESUME_LINES, table_rows=[["Python", "Terraform"]])
        text = extract_text(content, ResumeFileType.DOCX)
        assert "Senior Software Engineer at Acme Health" in text
        assert "Python | Terraform" in text
        assert "Led migration" in text and "- Led" not in text  # bullet stripped

    def test_pdf(self, sample_pdf: bytes) -> None:
        text = extract_text(sample_pdf, ResumeFileType.PDF)
        assert "Acme Health" in text and "Kubernetes" in text

    def test_too_little_text(self) -> None:
        with pytest.raises(TextExtractionError, match="too little"):
            extract_text(make_docx(["Hi"]), ResumeFileType.DOCX)

    def test_corrupt_file(self) -> None:
        with pytest.raises(TextExtractionError, match="could not read"):
            extract_text(b"%PDF-1.4 garbage", ResumeFileType.PDF)

    def test_normalize(self) -> None:
        raw = "A B\r\n\r\n\r\n• ﬁnance\x07  team\n\n"
        assert normalize_text(raw) == "A B\n\nfinance team"


# --------------------------------------------------------------------------- heuristic


class TestHeuristicExtractor:
    def test_extracts_sections(self) -> None:
        p = HeuristicResumeExtractor().extract(normalize_text(SAMPLE_TEXT))
        assert p.full_name == "Jane Doe"
        assert {"Python", "Go", "SQL", "FastAPI", "Django"} <= set(p.skills)
        assert {"AWS", "Docker", "Kubernetes", "PostgreSQL"} <= set(p.technologies)
        assert p.certifications == ["AWS Certified Solutions Architect - Associate"]
        assert [pos.title for pos in p.positions] == [
            "Senior Software Engineer",
            "Software Engineer",
        ]
        senior = p.positions[0]
        assert senior.company == "Acme Health"
        assert senior.start_date == date(2020, 1, 1) and senior.is_current
        assert p.positions[1].end_date == date(2019, 12, 1)
        assert p.education[0].graduation_year == 2016
        assert p.education[0].degree is not None and "Computer Science" in p.education[0].degree


# --------------------------------------------------------------------------- grounding


class TestGrounding:
    def test_fabricated_items_are_dropped(self) -> None:
        fabricated = ResumeProfile(
            full_name="Jane Doe",
            skills=["Python", "Rust", "C++"],  # Rust / C++ are not in the resume
            technologies=["Kubernetes", "Kafka"],
            certifications=[
                "AWS Certified Solutions Architect - Associate",
                "Certified Kubernetes Administrator",
            ],
            job_titles=["Senior Software Engineer", "Engineering Manager"],
            positions=[
                Position(
                    title="Senior Software Engineer",
                    company="Acme Health",
                    start_date=date(2020, 1, 1),
                    is_current=True,
                    highlights=[
                        "Led migration of billing services to Kubernetes",
                        "Grew revenue by 400% single-handedly",
                    ],
                ),
                Position(
                    title="Staff Engineer",
                    company="Google",
                    start_date=date(2012, 1, 1),
                    end_date=date(2015, 1, 1),
                ),
            ],
            education=[Education(institution="MIT", degree="Ph.D. Physics")],
            domains=["healthcare", "fintech"],
            accomplishments=["Won a Nobel prize"],
        )
        grounded, warnings = ground_profile(fabricated, SAMPLE_TEXT, today=TODAY)
        assert grounded.skills == ["Python"]
        assert grounded.technologies == ["Kubernetes"]
        assert grounded.certifications == ["AWS Certified Solutions Architect - Associate"]
        assert grounded.job_titles == ["Senior Software Engineer"]
        assert [p.title for p in grounded.positions] == ["Senior Software Engineer"]
        assert grounded.positions[0].highlights == [
            "Led migration of billing services to Kubernetes"
        ]
        assert grounded.education == []
        assert grounded.domains == ["healthcare"]
        assert grounded.accomplishments == []
        assert len(warnings) >= 10
        assert any("Rust" in w for w in warnings)

    def test_skill_matching_respects_word_boundaries(self) -> None:
        text = "Experienced with Gopher tools and Javascript and C# and Node.js"
        grounded, _ = ground_profile(
            ResumeProfile(skills=["Go", "Java", "C#", "Node.js", "C"]), text, today=TODAY
        )
        assert grounded.skills == ["C#", "Node.js"]

    def test_unverifiable_dates_dropped(self) -> None:
        profile = ResumeProfile(
            positions=[
                Position(
                    title="Software Engineer",
                    start_date=date(2011, 1, 1),
                    end_date=date(2013, 1, 1),
                )
            ]
        )
        grounded, warnings = ground_profile(profile, SAMPLE_TEXT, today=TODAY)
        assert grounded.positions[0].start_date is None
        assert grounded.years_of_experience is None
        assert any("unverifiable dates" in w for w in warnings)

    def test_years_recomputed_from_dates(self) -> None:
        profile = HeuristicResumeExtractor().extract(normalize_text(SAMPLE_TEXT))
        profile.years_of_experience = 25
        grounded, warnings = ground_profile(profile, SAMPLE_TEXT, today=TODAY)
        assert grounded.years_of_experience == pytest.approx(10.2, abs=0.2)
        assert any("replaced by" in w for w in warnings)

    def test_claimed_years_without_dates(self) -> None:
        text = "Engineer with 8+ years of experience building APIs in Python and SQL."
        kept, _ = ground_profile(ResumeProfile(years_of_experience=8), text, today=TODAY)
        assert kept.years_of_experience == 8
        dropped, w = ground_profile(ResumeProfile(years_of_experience=12), text, today=TODAY)
        assert dropped.years_of_experience is None and w


class TestExperienceYears:
    def test_overlaps_merged(self) -> None:
        positions = [
            Position(title="A", start_date=date(2020, 1, 1), end_date=date(2022, 1, 1)),
            Position(title="B", start_date=date(2021, 1, 1), end_date=date(2023, 1, 1)),
            Position(title="C", start_date=date(2024, 1, 1), is_current=True),
        ]
        assert compute_experience_years(positions, date(2025, 1, 1)) == pytest.approx(4.0, 0.01)

    def test_no_dates(self) -> None:
        assert compute_experience_years([Position(title="A")], TODAY) is None


# --------------------------------------------------------------------------- pipeline


class TestResumeParser:
    def test_docx_end_to_end(self, sample_docx: bytes) -> None:
        parsed = ResumeParser().parse("jane.docx", sample_docx, today=TODAY)
        assert parsed.file_type is ResumeFileType.DOCX
        assert len(parsed.content_sha256) == 64
        assert "Kubernetes" in parsed.profile.technologies
        assert parsed.profile.years_of_experience is not None

    def test_pdf_end_to_end(self, sample_pdf: bytes) -> None:
        parsed = ResumeParser().parse("jane.pdf", sample_pdf, today=TODAY)
        assert parsed.profile.full_name == "Jane Doe"
        assert "Python" in parsed.profile.skills

    def test_falls_back_when_primary_fails(self, sample_docx: bytes) -> None:
        parsed = ResumeParser(FailingExtractor()).parse("x.docx", sample_docx, today=TODAY)
        assert "Python" in parsed.profile.skills
        assert any("fallback" in w for w in parsed.warnings)

    def test_primary_output_still_grounded(self, sample_docx: bytes) -> None:
        extractor = StaticExtractor(ResumeProfile(skills=["Python", "COBOL"]))
        parsed = ResumeParser(extractor).parse("x.docx", sample_docx, today=TODAY)
        assert parsed.profile.skills == ["Python"]


class TestLLMExtractor:
    def test_structured_output_returned(self) -> None:
        model = FakeStructuredModel(ResumeProfile(skills=["Python"]))
        result = LLMResumeExtractor(model).extract("resume text")  # type: ignore[arg-type]
        assert result.skills == ["Python"]
        messages = model.prompts[0].to_messages()
        assert "Never infer" in messages[0].content
        assert "<resume>\nresume text\n</resume>" in messages[1].content

    def test_dict_output_validated(self) -> None:
        model = FakeStructuredModel({"skills": ["Go"]})
        assert LLMResumeExtractor(model).extract("x").skills == ["Go"]  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "response", [{"skills": "not-a-list", "bogus": 1}, "text", TimeoutError("slow")]
    )
    def test_malformed_output_raises(self, response: Any) -> None:
        with pytest.raises(ExtractionFailed):
            LLMResumeExtractor(FakeStructuredModel(response)).extract("x")  # type: ignore[arg-type]

    def test_text_truncated(self) -> None:
        model = FakeStructuredModel(ResumeProfile())
        LLMResumeExtractor(model, max_chars=10).extract("x" * 100)  # type: ignore[arg-type]
        assert "x" * 11 not in model.prompts[0].to_messages()[1].content
