"""
streamlit_app.py
-----------------
Phase 2 deliverable: BD-facing search/filter UI over the parsed candidate data.

Run locally:   streamlit run streamlit_app.py
Deploy (free): push this repo to GitHub -> https://share.streamlit.io ->
               "New app" -> point at this file. That gives a public share
               link (streamlit.app URL) without needing your own server.

Data source: reads output/parsed_resumes.json (falls back to .csv) produced
by src/llm_parser.py. Nothing in this file talks to the Anthropic API --
parsing is a separate, offline batch step (see the notebook) so the UI stays
fast and doesn't burn API spend on every filter click.
"""

from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Candidate Search — BD Sourcing",
    page_icon="🔎",
    layout="wide",
)

DATA_DIR = (Path(__file__).parent if "__file__" in globals() else Path.cwd()) / "output"


@st.cache_data
def load_candidates() -> pd.DataFrame:
    """
    Load the parsed candidate JSON (full-fidelity) and flatten it into a
    DataFrame the UI can filter on. Cached so re-running filters doesn't
    re-read/re-parse the file on every interaction.
    """
    json_path = DATA_DIR / "parsed_resumes.json"
    if not json_path.exists():
        st.error(
            f"Couldn't find {json_path}. Run the parsing notebook first "
            "(src/llm_parser.py -> output/parsed_resumes.json)."
        )
        st.stop()

    with open(json_path) as f:
        records = json.load(f)

    rows = []
    for r in records:
        rows.append(
            {
                "full_name": r["full_name"],
                "source_file": r["source_file"],
                "geography": r["geography"],
                "strategy_type": r["strategy_type"],
                "sectors": r.get("sectors", []),
                "seniority": r["seniority"],
                "total_years_experience": r["total_years_experience"],
                "current_employer": r.get("current_employer"),
                "current_title": r.get("current_title"),
                "education": r.get("education", []),
                "certifications": r.get("certifications", []),
                "technical_skills": r.get("technical_skills", []),
                "languages": r.get("languages", []),
                "work_history": r.get("work_history", []),
                "key_achievements": r.get("key_achievements", []),
                "parse_confidence": r.get("parse_confidence", "high"),
                "parse_notes": r.get("parse_notes"),
            }
        )
    return pd.DataFrame(rows)


df = load_candidates()

st.sidebar.title("🔎 Filters")
st.sidebar.caption(f"{len(df)} candidates in the database")

if st.sidebar.button("↺ Clear all filters"):
    for key in ("search_text", "geo_filter", "strategy_filter", "sector_filter",
                "seniority_filter", "yrs_range", "cert_filter"):
        st.session_state.pop(key, None)
    st.rerun()

search_text = st.sidebar.text_input(
    "Search (name, employer, skill, achievement)",
    placeholder="e.g. Goldman, Python, CFA...",
    key="search_text",
)

all_geo = sorted(df["geography"].unique())
geo_filter = st.sidebar.multiselect("Geography", all_geo, default=all_geo, key="geo_filter")

all_strategy = sorted(df["strategy_type"].unique())
strategy_filter = st.sidebar.multiselect(
    "Strategy Type", all_strategy, default=all_strategy, key="strategy_filter"
)

all_sectors = sorted({s for row in df["sectors"] for s in row})
sector_filter = st.sidebar.multiselect("Sector", all_sectors, default=[], key="sector_filter")

seniority_order = ["Intern/Analyst (0-2 yrs)", "Junior (2-4 yrs)", "Mid (4-7 yrs)", "Senior (7+ yrs)"]
all_seniority = [s for s in seniority_order if s in df["seniority"].unique()]
seniority_filter = st.sidebar.multiselect(
    "Seniority", all_seniority, default=all_seniority, key="seniority_filter"
)

max_yrs = float(df["total_years_experience"].max())
yrs_range = st.sidebar.slider(
    "Years of experience", min_value=0.0, max_value=max(max_yrs, 1.0),
    value=(0.0, max_yrs), step=0.5, key="yrs_range",
)

cert_filter = st.sidebar.checkbox(
    "Has a professional certification (CFA/CAIA/FRM/etc.)", key="cert_filter"
)

filtered = df[
    df["geography"].isin(geo_filter)
    & df["strategy_type"].isin(strategy_filter)
    & df["seniority"].isin(seniority_filter)
    & df["total_years_experience"].between(yrs_range[0], yrs_range[1])
]

if sector_filter:
    filtered = filtered[filtered["sectors"].apply(lambda s: any(sec in s for sec in sector_filter))]

if cert_filter:
    filtered = filtered[filtered["certifications"].apply(lambda c: len(c) > 0)]

if search_text:
    q = search_text.lower()

    def matches(row) -> bool:
        haystack = " ".join(
            [
                row["full_name"],
                str(row["current_employer"] or ""),
                str(row["current_title"] or ""),
                " ".join(row["technical_skills"]),
                " ".join(row["certifications"]),
                " ".join(row["key_achievements"]),
                " ".join(e["employer"] for e in row["work_history"]),
                " ".join(e["institution"] for e in row["education"]),
                " ".join(e["degree"] for e in row["education"]),
                " ".join(row["languages"]),
            ]
        ).lower()
        return q in haystack

    filtered = filtered[filtered.apply(matches, axis=1)]

st.title("Candidate Search — BD Sourcing Platform")
st.caption(
    "Search and filter parsed analyst candidate resumes across geography, "
    "strategy type, sector, and seniority."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Candidates matching filters", len(filtered))
m2.metric("Total candidates", len(df))
m3.metric(
    "Avg. years experience (filtered)",
    f"{filtered['total_years_experience'].mean():.1f}" if len(filtered) else "—",
)
m4.metric(
    "Systematic/Quant (filtered)",
    int((filtered["strategy_type"] == "Systematic/Quantitative").sum()) if not filtered.empty else 0,
)

if filtered.empty:
    st.warning("No candidates match the current filters. Try widening your filters above.")

tab_results, tab_insights = st.tabs(["📋 Candidates", "📊 Insights"])

with tab_results:
    if filtered.empty:
        st.warning("No candidates match the current filters.")
    else:
        sort_by = st.selectbox(
            "Sort by",
            ["Years of experience (high to low)", "Years of experience (low to high)", "Name (A-Z)"],
        )
        if sort_by.startswith("Years") and "high" in sort_by:
            filtered = filtered.sort_values("total_years_experience", ascending=False)
        elif sort_by.startswith("Years"):
            filtered = filtered.sort_values("total_years_experience", ascending=True)
        else:
            filtered = filtered.sort_values("full_name")

        for _, c in filtered.iterrows():
            with st.expander(
                f"**{c['full_name']}** — {c['current_title'] or 'N/A'} @ "
                f"{c['current_employer'] or 'N/A'}  ·  {c['total_years_experience']:.1f} yrs  ·  "
                f"{c['geography']}  ·  {c['strategy_type']}"
            ):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Sectors:** {', '.join(c['sectors']) or '—'}")
                    st.markdown(f"**Seniority:** {c['seniority']}")
                    st.markdown(f"**Certifications:** {', '.join(c['certifications']) or '—'}")
                    st.markdown(f"**Technical skills:** {', '.join(c['technical_skills']) or '—'}")
                    if c["key_achievements"]:
                        st.markdown("**Key achievements:**")
                        for a in c["key_achievements"]:
                            st.markdown(f"- {a}")
                    if c["work_history"]:
                        st.markdown("**Work history:**")
                        for job in c["work_history"]:
                            dates = f"{job.get('start_date') or '?'} – {job.get('end_date') or '?'}"
                            st.markdown(f"- **{job['title']}**, {job['employer']} ({dates})")
                with col2:
                    if c["education"]:
                        st.markdown("**Education:**")
                        for e in c["education"]:
                            st.markdown(f"- {e['degree']}, {e['institution']} ({e.get('graduation_year') or '?'})")
                    if c["parse_confidence"] != "high":
                        st.caption(f"⚠️ Parse confidence: {c['parse_confidence']}")
                        if c["parse_notes"]:
                            st.caption(c["parse_notes"])

        export_df = filtered.copy()
        export_df["sectors"] = export_df["sectors"].apply(lambda v: "; ".join(v))
        export_df["certifications"] = export_df["certifications"].apply(lambda v: "; ".join(v))
        export_df["technical_skills"] = export_df["technical_skills"].apply(lambda v: "; ".join(v))
        export_df["languages"] = export_df["languages"].apply(lambda v: "; ".join(v))
        export_df["education"] = export_df["education"].apply(
            lambda v: "; ".join(f"{e['degree']} @ {e['institution']}" for e in v)
        )
        export_df = export_df.drop(columns=["work_history", "key_achievements"])

        st.download_button(
            "⬇️ Download filtered results (CSV)",
            export_df.to_csv(index=False),
            file_name="filtered_candidates.csv",
            mime="text/csv",
        )

with tab_insights:
    scope = st.radio(
        "Show insights for:",
        ["All candidates", "Filtered candidates only"],
        horizontal=True,
        key="insights_scope",
    )
    chart_df = df if scope == "All candidates" else filtered

    if chart_df.empty:
        st.warning("No candidates to show — your current filters return zero results.")
    else:
        st.subheader(
            "Candidate pool distribution (all candidates)"
            if scope == "All candidates"
            else f"Candidate pool distribution ({len(chart_df)} filtered candidates)"
        )

        c1, c2 = st.columns(2)
        with c1:
            geo_counts = chart_df["geography"].value_counts().reset_index()
            geo_counts.columns = ["geography", "count"]
            fig = px.bar(geo_counts, x="geography", y="count", title="Candidates by Geography")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            strat_counts = chart_df["strategy_type"].value_counts().reset_index()
            strat_counts.columns = ["strategy_type", "count"]
            fig = px.pie(strat_counts, names="strategy_type", values="count", title="Fundamental vs. Systematic")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            sector_counts = pd.Series(
                [s for row in chart_df["sectors"] for s in row]
            ).value_counts().reset_index()
            sector_counts.columns = ["sector", "count"]
            fig = px.bar(sector_counts, x="sector", y="count", title="Candidates by Sector Coverage")
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            fig = px.histogram(
                chart_df, x="total_years_experience", nbins=8,
                title="Years of Experience Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Seniority mix")
        sen_counts = chart_df["seniority"].value_counts().reindex(seniority_order).dropna().reset_index()
        sen_counts.columns = ["seniority", "count"]
        fig = px.bar(sen_counts, x="seniority", y="count", title="Candidates by Seniority Band")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Data parsed from resume PDFs/DOCX via the Anthropic API (see the notebook's "
    "src/llm_parser.py). Built for a 10-candidate sample — see notebook Section 8 "
    "for how this scales to thousands of resumes."
)
