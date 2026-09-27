"""
Apply Assistant — agentic job-application agent
LangGraph skeleton: Planner -> JD Parser -> Skill Matcher -> Gap Checker
                     -> Drafter -> Reviewer

Fill in GEMINI_API_KEY and a search tool call before the demo.
Each node returns a partial state update; LangGraph merges it into `AgentState`.
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")


# ---------- State ----------

class AgentState(TypedDict):
    jd_text: str
    candidate_profile: dict          # seeded JSON: skills, projects, experience
    trace: List[str]                 # human-readable log of each step, shown live in UI
    company_notes: Optional[str]
    jd_requirements: Optional[List[str]]
    matched_skills: Optional[List[str]]
    gaps: Optional[List[str]]
    draft: Optional[str]
    review_notes: Optional[str]
    final_output: Optional[str]


def log(state: AgentState, message: str) -> List[str]:
    return state["trace"] + [message]


# ---------- Node prompts ----------

PLANNER_PROMPT = """You are planning the steps needed to tailor a job application.
Given this job description, list in one short sentence what information you still
need to gather before drafting (e.g. company context, required skills, candidate fit).
Do not draft anything yet — just state the plan.

Job description:
{jd_text}
"""

JD_PARSER_PROMPT = """Extract the required and preferred skills/keywords from this job
description as a plain comma-separated list. Only include concrete skills, tools, or
qualifications — no soft skills, no filler.

Job description:
{jd_text}
"""

SKILL_MATCHER_PROMPT = """Compare the candidate's real skills against the job's required
skills. Return two comma-separated lists on separate lines:
MATCHED: skills the candidate genuinely has that overlap with the job requirements
GAPS: job requirements the candidate does NOT have

Candidate skills/projects:
{candidate_profile}

Job requirements:
{jd_requirements}
"""

DRAFTER_PROMPT = """Write 3-4 tailored resume bullet points for this candidate applying
to this job. Use ONLY the matched skills listed below — never invent or imply experience
the candidate doesn't have. Be specific and quantify impact where the candidate profile
supports it.

Matched skills: {matched_skills}
Candidate background: {candidate_profile}
Job description: {jd_text}
"""

REVIEWER_PROMPT = """Review this draft against the candidate's real profile. Flag any
claim that is not directly supported by the candidate's background. If everything is
honest and well-supported, say "APPROVED". Otherwise list the specific unsupported
claims to remove.

Draft:
{draft}

Candidate's real profile:
{candidate_profile}
"""


# ---------- Nodes ----------

def planner_node(state: AgentState) -> dict:
    resp = model.generate_content(PLANNER_PROMPT.format(jd_text=state["jd_text"]))
    return {"trace": log(state, f"Planner: {resp.text.strip()}")}


def company_research_node(state: AgentState) -> dict:
    # Swap in a real search tool call here (Tavily/SerpAPI). Placeholder for now.
    notes = "Company research placeholder — wire up a search tool before the demo."
    return {"company_notes": notes, "trace": log(state, f"Company research: {notes}")}


def jd_parser_node(state: AgentState) -> dict:
    resp = model.generate_content(JD_PARSER_PROMPT.format(jd_text=state["jd_text"]))
    reqs = [r.strip() for r in resp.text.split(",") if r.strip()]
    return {"jd_requirements": reqs, "trace": log(state, f"Parsed requirements: {reqs}")}


def skill_matcher_node(state: AgentState) -> dict:
    resp = model.generate_content(SKILL_MATCHER_PROMPT.format(
        candidate_profile=state["candidate_profile"],
        jd_requirements=state["jd_requirements"],
    ))
    text = resp.text
    matched, gaps = [], []
    for line in text.splitlines():
        if line.upper().startswith("MATCHED:"):
            matched = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
        elif line.upper().startswith("GAPS:"):
            gaps = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
    return {
        "matched_skills": matched,
        "gaps": gaps,
        "trace": log(state, f"Matched: {matched} | Gaps: {gaps}"),
    }


def drafter_node(state: AgentState) -> dict:
    resp = model.generate_content(DRAFTER_PROMPT.format(
        matched_skills=state["matched_skills"],
        candidate_profile=state["candidate_profile"],
        jd_text=state["jd_text"],
    ))
    return {"draft": resp.text.strip(), "trace": log(state, "Draft generated")}


def reviewer_node(state: AgentState) -> dict:
    resp = model.generate_content(REVIEWER_PROMPT.format(
        draft=state["draft"],
        candidate_profile=state["candidate_profile"],
    ))
    review = resp.text.strip()
    approved = review.strip().upper().startswith("APPROVED")
    final = state["draft"] if approved else f"{state['draft']}\n\n[Reviewer flagged]: {review}"
    return {
        "review_notes": review,
        "final_output": final,
        "trace": log(state, f"Reviewer: {'approved' if approved else 'flagged issues'}"),
    }


# ---------- Graph ----------

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("company_research", company_research_node)
    graph.add_node("jd_parser", jd_parser_node)
    graph.add_node("skill_matcher", skill_matcher_node)
    graph.add_node("drafter", drafter_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "company_research")
    graph.add_edge("company_research", "jd_parser")
    graph.add_edge("jd_parser", "skill_matcher")
    graph.add_edge("skill_matcher", "drafter")
    graph.add_edge("drafter", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "jd_text": "Paste a job description here for a quick CLI test.",
        "candidate_profile": {
            "skills": ["Python", "LangChain", "FAISS", "Gemini API", "RAG", "Streamlit"],
            "projects": ["RAG-based tutoring chatbot (NTCC project)"],
        },
        "trace": [],
    })
    for step in result["trace"]:
        print(step)
    print("\n--- FINAL OUTPUT ---\n")
    print(result["final_output"])
