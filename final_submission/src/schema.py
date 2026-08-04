"""
schema.py
---------
Defines the structured record every resume gets parsed into.

Field choices map directly to the BD team's stated filter dimensions:
  - Geographic Markets:      candidate.geography
  - Investment Approach:     candidate.strategy_type
  - Sector:                  candidate.sectors (multi-value -- most people
                              cover more than one)
  - Experience Level:        candidate.total_years_experience,
                              candidate.seniority

Everything else (education, certifications, skills, employer history) is
kept because BD users will want to search/filter on it too (e.g. "CFA
holders", "ex-Goldman", "Python + ML skills" for systematic roles), and
because it's what a human recruiter would read next after the headline
filters narrow the list.

Using Pydantic gives us for free:
  - a JSON-schema we can hand to the LLM so its output is constrained
  - validation on the way back (catches hallucinated fields/types early)
  - a single source of truth for the CSV / Streamlit column list later
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Education(BaseModel):
    degree: str
    institution: str
    field_of_study: Optional[str] = None
    graduation_year: Optional[str] = None


class WorkExperience(BaseModel):
    employer: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    sectors_covered: list[str] = Field(default_factory=list)
    description: str = ""


class CandidateProfile(BaseModel):
    source_file: str
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    current_location: Optional[str] = None

    geography: Literal["US", "Europe", "Asia-Pacific", "Multiple/Global"] = Field(
        description="One of: US, Europe, Asia-Pacific, Multiple/Global"
    )
    strategy_type: Literal["Fundamental", "Systematic/Quantitative", "Both", "Unclear"] = Field(
        description="One of: Fundamental, Systematic/Quantitative, Both, Unclear"
    )
    sectors: list[str] = Field(
        default_factory=list,
        description="e.g. Technology, Healthcare, Financial Services, Energy, "
                    "Industrials, Consumer, Credit, Macro, Generalist",
    )
    seniority: Literal[
        "Intern/Analyst (0-2 yrs)", "Junior (2-4 yrs)", "Mid (4-7 yrs)", "Senior (7+ yrs)"
    ] = Field(
        description="One of: Intern/Analyst (0-2 yrs), Junior (2-4 yrs), "
                    "Mid (4-7 yrs), Senior (7+ yrs)"
    )
    total_years_experience: float

    current_employer: Optional[str] = None
    current_title: Optional[str] = None

    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    technical_skills: list[str] = Field(
        default_factory=list,
        description="Python, R, SQL, Bloomberg, FactSet, machine learning, C++, etc.",
    )
    languages: list[str] = Field(default_factory=list)
    work_history: list[WorkExperience] = Field(default_factory=list)
    key_achievements: list[str] = Field(
        default_factory=list,
        description="3-6 standout, quantifiable bullet points across the whole career",
    )

    parse_confidence: str = Field(
        default="high", description="high / medium / low -- LLM's own confidence flag"
    )
    parse_notes: Optional[str] = Field(
        default=None, description="Anything ambiguous the LLM flagged for human review"
    )


CANDIDATE_JSON_SCHEMA = CandidateProfile.model_json_schema()
