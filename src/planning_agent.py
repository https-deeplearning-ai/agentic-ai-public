import json
import re
import textwrap
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from src.agents import (
    research_agent,
    writer_agent,
    editor_agent,
)
from src.llm_client import get_aisuite_client
from src.research_tools import tavily_search_tool, arxiv_search_tool

"""LLM configuration shims for OpenAI-compatible servers.

Prefer generic envs (LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL).
Normalize to OPENAI_* for clients that expect those names, then init Client.
"""
if os.getenv("LLM_BASE_URL") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("LLM_BASE_URL", "")
if os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("LLM_API_KEY", "")

DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai:gpt-4.1-mini")

client = get_aisuite_client()

TOOLLESS_PROVIDERS = {"ollama"}


def clean_json_block(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip("` \n")


from typing import List
import json, ast


def planner_agent(topic: str, model: Optional[str] = None) -> List[str]:
    prompt = f"""
You are a planning agent responsible for organizing a research workflow using multiple intelligent agents.

🧠 Available agents:
- Research agent: MUST begin with a broad **web search using Tavily** to identify only **relevant** and **authoritative** items (e.g., high-impact venues, seminal works, surveys, or recent comprehensive sources). The output of this step MUST capture for each candidate: title, authors, year, venue/source, URL, and (if available) DOI.
- Research agent: AFTER the Tavily step, perform a **targeted arXiv search** ONLY for the candidates discovered in the web step (match by title/author/DOI). If an arXiv preprint/version exists, record its arXiv URL and version info. Do NOT run a generic arXiv search detached from the Tavily results.
- Writer agent: drafts based on research findings.
- Editor agent: reviews, reflects on, and improves drafts.

🎯 Produce a clear step-by-step research plan **as a valid Python list of strings** (no markdown, no explanations). 
Each step must be atomic, actionable, and assigned to one of the agents.
Maximum of 7 steps.

🚫 DO NOT include steps like “create CSV”, “set up repo”, “install packages”.
✅ Focus on meaningful research tasks (search, extract, rank, draft, revise).
✅ The FIRST step MUST be exactly: 
"Research agent: Use Tavily to perform a broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
✅ The SECOND step MUST be exactly:
"Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."

🔚 The FINAL step MUST instruct the writer agent to generate a comprehensive Markdown report that:
- Uses all findings and outputs from previous steps
- Includes inline citations (e.g., [1], (Wikipedia/arXiv))
- Includes a References section with clickable links for all citations
- Preserves earlier sources
- Is detailed and self-contained

Topic: "{topic}"
"""

    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
    )

    raw = response.choices[0].message.content.strip()

    # --- robust parsing: JSON -> ast -> fallback ---
    def _coerce_to_list(s: str) -> List[str]:
        # try strict JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj[:7]
        except json.JSONDecodeError:
            pass
        # try Python literal list
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj[:7]
        except Exception:
            pass
        # try to extract code fence if present
        if s.startswith("```") and s.endswith("```"):
            inner = s.strip("`")
            try:
                obj = ast.literal_eval(inner)
                if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                    return obj[:7]
            except Exception:
                pass
        return []

    steps = _coerce_to_list(raw)

    # enforce ordering & minimal contract
    required_first = "Research agent: Use Tavily to perform a broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
    required_second = "Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."
    final_required = "Writer agent: Generate the final comprehensive Markdown report with inline citations and a complete References section with clickable links."

    def _ensure_contract(steps_list: List[str]) -> List[str]:
        if not steps_list:
            return [
                required_first,
                required_second,
                "Research agent: Synthesize and rank findings by relevance, recency, and authority; deduplicate by title/DOI.",
                "Writer agent: Draft a structured outline based on the ranked evidence.",
                "Editor agent: Review for coherence, coverage, and citation completeness; request fixes.",
                final_required,
            ]
        # inject/replace first two if missing or out of order
        steps_list = [s for s in steps_list if isinstance(s, str)]
        if not steps_list or steps_list[0] != required_first:
            steps_list = [required_first] + steps_list
        if len(steps_list) < 2 or steps_list[1] != required_second:
            # remove any generic arxiv step that is not tied to Tavily results
            steps_list = (
                [steps_list[0]]
                + [required_second]
                + [
                    s
                    for s in steps_list[1:]
                    if "arXiv" not in s or "For each collected item" in s
                ]
            )
        # ensure final step requirement present
        if final_required not in steps_list:
            steps_list.append(final_required)
        # cap to 7
        return steps_list[:7]

    steps = _ensure_contract(steps)

    return steps


def _provider_from_model(model_name: str) -> str:
    if not model_name:
        return ""
    if ":" in model_name:
        return model_name.split(":", 1)[0]
    return model_name


def _supports_tool_invocation(model_name: str) -> bool:
    provider = _provider_from_model(model_name)
    return provider not in TOOLLESS_PROVIDERS


def _format_tool_usage(entries: List[str]) -> str:
    if not entries:
        return ""

    unique = []
    for entry in entries:
        if entry not in unique:
            unique.append(entry)

    formatted = "\n".join(f"- {item}" for item in unique)
    return f"\n\n📎 Tools used\n{formatted}"


def _generate_search_query(model_name: str, user_prompt: str) -> str:
    instructions = (
        "You craft concise web search queries. Respond with a single query string "
        "(no bullet points) that would retrieve authoritative and timely sources."
    )
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                User research question:
                {user_prompt.strip()}
                """
            ).strip(),
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=48,
        )
        query = (resp.choices[0].message.content or "").strip()
        query = query.splitlines()[0].strip().strip('"')
        return query or user_prompt.strip()[:120]
    except Exception:
        fallback = user_prompt.strip()[:120]
        return fallback or "latest research developments"


def _run_tavily_fallback(
    user_prompt: str, model_name: str, shared_state: Dict[str, Any]
) -> str:
    query = _generate_search_query(model_name, user_prompt)
    try:
        results = tavily_search_tool(query=query, max_results=5)
    except Exception as exc:
        results = [{"error": f"Tavily call failed: {exc}"}]

    shared_state["tavily_seed_results"] = {
        "query": query,
        "results": results,
    }

    valid_results = [r for r in results if isinstance(r, dict) and not r.get("error")]
    errors = [r.get("error") for r in results if isinstance(r, dict) and r.get("error")]

    lines: List[str] = []
    lines.append("1. Summary of Research Approach:")
    lines.append(f"- Executed Tavily web search with query: {query}.")
    if valid_results:
        lines.append("- Collected the top relevant items with metadata for follow-up analysis.")
    lines.append("")

    lines.append("Key Findings:")
    if valid_results:
        for idx, item in enumerate(valid_results, start=1):
            bullet = chr(ord("A") + (idx - 1) % 26)
            title = item.get("title") or "(No title provided)"
            lines.append(f"{bullet}. {title}")
            snippet = (item.get("content") or "").strip()
            if snippet:
                lines.append(
                    textwrap.fill(
                        snippet,
                        width=96,
                        initial_indent="   ",
                        subsequent_indent="   ",
                    )
                )
            url = item.get("url")
            if url:
                lines.append(f"   URL: {url}")
    else:
        lines.append("- Tavily did not return any successful results for the generated query.")

    if errors:
        lines.append("")
        lines.append("Errors Encountered:")
        for err in errors:
            lines.append(f"- {err}")

    lines.append("")
    lines.append("Source Details:")
    if valid_results:
        for item in valid_results:
            title = item.get("title") or "(No title provided)"
            url = item.get("url") or "N/A"
            lines.append(f"- {title} — {url}")
    else:
        lines.append("- No source metadata available.")

    lines.append("")
    lines.append("Limitations:")
    lines.append("- Web snippets may omit deeper context or reference paywalled material.")
    lines.append("- Forecasts and claims about AGI vary widely across sources.")

    body = "\n".join(lines)
    body += _format_tool_usage(
        [f"tavily_search_tool(query={query!r}, max_results=5)"]
    )
    return body


def _run_arxiv_fallback(
    shared_state: Dict[str, Any],
) -> str:
    tavily_state = shared_state.get("tavily_seed_results") or {}
    tavily_results = tavily_state.get("results") or []
    query_used = tavily_state.get("query") or "<unknown>"

    valid_items = [
        item for item in tavily_results if isinstance(item, dict) and not item.get("error")
    ]
    if not valid_items:
        return (
            "No Tavily results are cached for arXiv cross-referencing. "
            "Re-run the Tavily step before executing the arXiv lookup."
        )

    matches: List[Dict[str, Any]] = []
    no_matches: List[str] = []
    errors: List[str] = []
    queries_run: List[str] = []

    for item in valid_items:
        title = (item.get("title") or "").strip()
        if not title:
            continue

        queries_run.append(title)
        try:
            hits = arxiv_search_tool(query=title, max_results=2)
        except Exception as exc:
            errors.append(f"{title}: {exc}")
            continue

        found_any = False
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            if hit.get("error"):
                errors.append(f"{title}: {hit['error']}")
                continue

            found_any = True
            matches.append(
                {
                    "query": title,
                    "title": hit.get("title"),
                    "url": hit.get("url"),
                    "link_pdf": hit.get("link_pdf"),
                    "authors": ", ".join(hit.get("authors", [])),
                    "published": hit.get("published"),
                }
            )

        if not found_any:
            no_matches.append(title)

    shared_state["arxiv_results"] = matches

    lines: List[str] = []
    lines.append("1. Summary of Research Approach:")
    lines.append(
        f"- Reviewed Tavily findings generated with query: {query_used}."
    )
    lines.append("- Queried arXiv for matching preprints (up to 2 per candidate title).")
    lines.append("")

    lines.append("Matches Found:")
    if matches:
        for idx, hit in enumerate(matches, start=1):
            title = hit.get("title") or "(Untitled)"
            lines.append(f"{idx}. {title}")
            meta_bits = []
            if hit.get("authors"):
                meta_bits.append(hit["authors"])
            if hit.get("published"):
                meta_bits.append(hit["published"])
            if meta_bits:
                lines.append(f"   {', '.join(meta_bits)}")
            if hit.get("url"):
                lines.append(f"   Abs: {hit['url']}")
            if hit.get("link_pdf"):
                lines.append(f"   PDF: {hit['link_pdf']}")
    else:
        lines.append("- No arXiv entries matched the Tavily candidates.")

    if no_matches:
        lines.append("")
        lines.append("Items Without arXiv Matches:")
        for title in no_matches:
            lines.append(f"- {title}")

    if errors:
        lines.append("")
        lines.append("Errors Encountered:")
        for err in errors:
            lines.append(f"- {err}")

    lines.append("")
    lines.append("Limitations:")
    lines.append("- arXiv covers scientific domains; industry or non-academic sources may be absent.")
    lines.append("- Matching relies on title similarity and may miss alternate phrasing or author listings.")

    body = "\n".join(lines)
    tool_entries = [
        f"arxiv_search_tool(query={query!r}, max_results=2)" for query in queries_run
    ]
    body += _format_tool_usage(tool_entries)
    return body


def executor_agent_step(
    step_title: str,
    history: list,
    prompt: str,
    model: Optional[str] = None,
    shared_state: Optional[Dict[str, Any]] = None,
):
    """
    Executes a step of the executor agent.
    Returns:
        - step_title (str)
        - agent_name (str)
        - output (str)
    """

    if shared_state is None:
        shared_state = {}

    model_name = model or DEFAULT_MODEL
    supports_tools = _supports_tool_invocation(model_name)

    # Construir contexto enriquecido estructurado
    context = f"📘 User Prompt:\n{prompt}\n\n📜 History so far:\n"
    for i, (desc, agent, output) in enumerate(history):
        if "draft" in desc.lower() or agent == "writer_agent":
            context += f"\n✍️ Draft (Step {i + 1}):\n{output.strip()}\n"
        elif "feedback" in desc.lower() or agent == "editor_agent":
            context += f"\n🧠 Feedback (Step {i + 1}):\n{output.strip()}\n"
        elif "research" in desc.lower() or agent == "research_agent":
            context += f"\n🔍 Research (Step {i + 1}):\n{output.strip()}\n"
        else:
            context += f"\n🧩 Other (Step {i + 1}) by {agent}:\n{output.strip()}\n"

    enriched_task = f"""{context}

🧩 Your next task:
{step_title}
"""

    # Seleccionar agente basado en el paso
    step_lower = step_title.lower()
    if "research" in step_lower:
        if not supports_tools and "use tavily" in step_lower:
            content = _run_tavily_fallback(prompt, model_name, shared_state)
            print("🔍 Research Agent (fallback Tavily) Output:", content)
            return step_title, "research_agent", content
        if not supports_tools and "arxiv" in step_lower:
            content = _run_arxiv_fallback(shared_state)
            print("🔍 Research Agent (fallback arXiv) Output:", content)
            return step_title, "research_agent", content

        content, _ = research_agent(prompt=enriched_task, model=model)
        print("🔍 Research Agent Output:", content)
        return step_title, "research_agent", content
    elif "draft" in step_lower or "write" in step_lower:
        content, _ = writer_agent(prompt=enriched_task, model=model)
        return step_title, "writer_agent", content
    elif "revise" in step_lower or "edit" in step_lower or "feedback" in step_lower:
        content, _ = editor_agent(prompt=enriched_task, model=model)
        return step_title, "editor_agent", content
    else:
        raise ValueError(f"Unknown step type: {step_title}")
