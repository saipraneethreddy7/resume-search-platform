"""
llm_parser.py
-------------
Phase 1 core: turns raw resume text into a validated CandidateProfile using
an LLM call (Anthropic Claude). This is the piece the case study explicitly
asks for -- "Parse resume data from PDF/Word documents using LLM models via API".

Design choices, and why:
  - Structured JSON output is enforced via Claude's tool-use / forced tool-choice
    mechanism (not just "please output JSON" in the prompt). This is far more
    reliable than asking for JSON in prose and regex-extracting it.
  - The tool's input_schema is generated directly from the Pydantic model in
    schema.py, so the model and the prompt can never drift apart.
  - Every response is re-validated against the Pydantic model before being
    accepted. A parse that fails validation is retried once with the
    validation error appended to the prompt (a common, cheap way to fix
    transient LLM formatting slips) before being logged as a failure.
  - Rate limiting / retry-with-backoff is included since Phase running over
    hundreds or thousands of resumes will hit rate limits.

To run this for real: `pip install anthropic`, set ANTHROPIC_API_KEY, then
call `parse_resume_batch(...)`. Swap `MODEL` for a cheaper/faster model
if throughput matters more than accuracy at scale.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from schema import CandidateProfile, CANDIDATE_JSON_SCHEMA

MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 2

SYSTEM_PROMPT = """You are a resume-parsing assistant for a hedge fund's Business \
Development team that sources junior/mid-level analyst talent across fundamental \
equity, systematic/quantitative, and credit strategies, across US, Europe, and \
Asia-Pacific markets.

Extract structured data from the resume text into the `record_candidate` tool. Rules:
- Only use information present in the resume. Never invent employers, dates, or numbers.
- geography: infer from where the candidate has worked/studied, not just their name origin.
- strategy_type: "Systematic/Quantitative" requires clear evidence (e.g. multi-factor \
models, Python/R-based backtesting, quant research roles). Sell-side/buy-side \
fundamental equity/credit research defaults to "Fundamental". Use "Both" only when \
the resume shows real split evidence, and "Unclear" when there's not enough signal.
- sectors: use the BD team's sector taxonomy (Technology, Healthcare, Financial \
Services, Energy, Industrials, Consumer, Credit, Macro) plus "Generalist" if the \
candidate covers many sectors without one clear focus.
- total_years_experience: sum full-time, post-degree work experience (exclude \
internships unless that is genuinely the person's only experience).
- key_achievements: prefer quantified bullets (%, $, bps, deal count) over vague ones.
- If something is ambiguous, still make your best-supported call, but set \
parse_confidence to "medium" or "low" and explain briefly in parse_notes.
"""

TOOL_DEFINITION = {
    "name": "record_candidate",
    "description": "Record one candidate's structured profile extracted from their resume.",
    "input_schema": CANDIDATE_JSON_SCHEMA,
}


def _build_user_message(filename: str, resume_text: str) -> str:
    return (
        f"Resume file: {filename}\n\n"
        f"--- RAW RESUME TEXT ---\n{resume_text}\n--- END RESUME TEXT ---\n\n"
        "Call record_candidate with the extracted fields."
    )


def parse_resume(filename: str, resume_text: str, client=None) -> Optional[CandidateProfile]:
    """
    Parse a single resume via the Anthropic API.
    `client` is an anthropic.Anthropic() instance, injected so this is testable
    without hitting the network, and so a single client/connection pool is
    reused across a batch instead of reconnecting per file.
    """
    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    last_error = None
    raw = None
    messages = [{"role": "user", "content": _build_user_message(filename, resume_text)}]

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                tools=[TOOL_DEFINITION],
                tool_choice={"type": "tool", "name": "record_candidate"},
                messages=messages,
            )
            tool_use_block = next(
                b for b in response.content if b.type == "tool_use"
            )
            raw = tool_use_block.input
            raw["source_file"] = filename
            return CandidateProfile.model_validate(raw)

        except ValidationError as e:
            last_error = e
            messages.append({"role": "assistant", "content": str(raw)})
            messages.append(
                {
                    "role": "user",
                    "content": f"That output failed validation:\n{e}\nPlease call "
                    "record_candidate again with corrected fields.",
                }
            )
        except Exception as e:
            last_error = e
            is_auth_error = (
                getattr(e, "status_code", None) == 401
                or "authenticat" in str(e).lower()
                or "api key" in str(e).lower()
                or e.__class__.__name__ == "AuthenticationError"
            )
            if is_auth_error:
                print(f"[llm_parser] AUTH ERROR on {filename}, not retrying: {e}")
                return None
            time.sleep(2 ** attempt)

    print(f"[llm_parser] FAILED after retries on {filename}: {last_error}")
    return None


def parse_resume_batch(
    resume_texts: dict[str, str], client=None
) -> tuple[list[CandidateProfile], list[str]]:
    """
    Parse every resume in `resume_texts` ({filename: text}).
    Returns (successes, failed_filenames). Sequential by default for
    simplicity/rate-limit safety; see notes.md for how this parallelizes
    at scale (async client + semaphore, or a queue + worker pool).
    """
    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    successes, failures = [], []
    total = len(resume_texts)
    for i, (filename, text) in enumerate(resume_texts.items(), start=1):
        profile = parse_resume(filename, text, client=client)
        if profile is not None:
            successes.append(profile)
            print(
                f"✓ Parsed {i}/{total} — {profile.full_name} "
                f"({profile.parse_confidence} confidence)"
            )
        else:
            failures.append(filename)
            print(f"✗ Failed {i}/{total} — {filename}")
    return successes, failures


def save_outputs(profiles: list[CandidateProfile], out_dir: Path) -> None:
    """Write parsed profiles to both JSON (full fidelity) and CSV (flat, for quick filtering)."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "parsed_resumes.json"
    with open(json_path, "w") as f:
        json.dump([p.model_dump() for p in profiles], f, indent=2)

    rows = []
    for p in profiles:
        rows.append(
            {
                "source_file": p.source_file,
                "full_name": p.full_name,
                "email": p.email,
                "geography": p.geography,
                "strategy_type": p.strategy_type,
                "sectors": ", ".join(p.sectors),
                "seniority": p.seniority,
                "total_years_experience": p.total_years_experience,
                "current_employer": p.current_employer,
                "current_title": p.current_title,
                "certifications": ", ".join(p.certifications),
                "technical_skills": ", ".join(p.technical_skills),
                "num_employers": len(p.work_history),
                "parse_confidence": p.parse_confidence,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = out_dir / "parsed_resumes.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(profiles)} profiles -> {json_path.name}, {csv_path.name}")
