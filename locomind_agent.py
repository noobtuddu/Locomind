"""
Locomind - Autonomous Blog Writing Agent (Python)

Capstone-ready single-file implementation suitable for a Kaggle project or as a demo for the
Google x Kaggle 5-day AI Agent course.

Features:
- Multi-stage agentic pipeline: topic understanding -> research -> outline -> draft -> refine -> SEO
- Uses OpenAI's ChatCompletion (or Completion) API for generation. Set OPENAI_API_KEY in env.
- Optional Wikipedia research to ground facts (falls back to model-only if not available).
- CLI and function-based so you can integrate in notebooks or pipelines.

Requirements (pip):
  pip install openai wikipedia requests

Set environment variable:
  export OPENAI_API_KEY="sk-..."

Usage:
  python locomind_agent.py --topic "best pre-workout foods" --tone conversational --length long

Note:
- This is an educational capstone implementation. For production, add caching, rate-limit handling,
  robust web-research, factual validation, and plagiarism checks.

"""

import os
import time
import json
import argparse
from typing import List, Dict, Optional

import google.generativeai as genai

# ------------------------- Configuration -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable before running.")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-1.5-pro"  # upgraded to pro model for best reasoning + writing quality

# Tunable parameters
DEFAULT_TEMPERATURE = 0.4  # slightly higher for more human-like writing = 0.3
OUTLINE_MAX_SECTIONS = 8

# ------------------------- Utility helpers -------------------------

def call_chat_model(system: str, messages: List[Dict], temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = 1500):
    """Call Gemini model and return generated text."""
    try:
        prompt = []
        if system:
            prompt.append({"role": "system", "content": system})
        prompt.extend(messages)

        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")


# ------------------------- Agent stages (Optimized for best results) -------------------------
# Improvements:
# - Added richer briefing
# - More detailed research synthesis
# - Stronger outline logic
# - More human-style drafting
# - Multi-pass refinement for maximum quality

def understand_topic(topic: str, tone: str = "neutral") -> Dict:
    """Produce a short brief about the topic: intent, target audience, angle suggestions."""
    system = "You are Locomind, an expert blog-writing AI that produces structured briefs."
    user = f"Create a short research brief for the topic: '{topic}'. Suggest 3 possible angles and the ideal target audience. Keep it concise (<= 180 words) and include keywords. Tone: {tone}."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    text = call_chat_model(system, messages, temperature=0.2, max_tokens=400)
    return {"brief": text}


def web_research_wikipedia(topic: str, sentences: int = 6) -> str:
    """Fetch a short research snippet from Wikipedia if available. Returns empty string on failure."""
    if not WIKI_AVAILABLE:
        return ""
    try:
        wikipedia.set_lang("en")
        search_results = wikipedia.search(topic, results=5)
        if not search_results:
            return ""
        page = wikipedia.page(search_results[0], auto_suggest=False)
        summary = wikipedia.summary(page.title, sentences=sentences)
        return f"From Wikipedia ({page.title}):\n" + summary
    except Exception:
        return ""


def build_outline(topic: str, brief_text: str, tone: str = "neutral", max_sections: int = OUTLINE_MAX_SECTIONS) -> Dict:
    system = "You are Locomind. Produce an SEO-friendly, logical blog outline with sections and short notes for each section."
    user = (
        f"Topic: {topic}\nBrief: {brief_text}\n\nCreate an ordered outline with {max_sections} sections (including introduction and conclusion). "
        "For each section provide a 1-line description of what to cover and 3 suggested sub-points. Make it SEO-aware."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    outline_text = call_chat_model(system, messages, temperature=0.2, max_tokens=700)
    return {"outline_text": outline_text}


def generate_draft(topic: str, outline_text: str, research_text: str, tone: str = "neutral", length: str = "medium") -> str:
    """Generate a first full draft based on the outline and optional research."""
    len_map = {"short": 400, "medium": 800, "long": 1500}
    target_tokens = len_map.get(length, 800)

    system = "You are Locomind, an expert long-form writer. Produce a coherent, well-structured blog article."
    user = (
        f"Topic: {topic}\nTone: {tone}\nLength target (approx. words): {target_tokens}\n\nResearch: {research_text}\n\nOutline:\n{outline_text}\n\nNow write a complete blog post based on the outline and research. Use headings for sections, include examples where relevant, and ensure smooth transitions. Keep it original and human-like."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    draft_text = call_chat_model(system, messages, temperature=0.35, max_tokens=2000)
    return draft_text


def refine_draft(draft: str, constraints: Optional[Dict] = None) -> str:
    """Multi-pass refinement for best possible quality."""
    constraints = constraints or {}

    system = "You are Locomind, a senior editor who rewrites text to sound human, intentional, and elegant."

    # Pass 1 — clarity
    p1_msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Rewrite the draft with perfect clarity, flow, and structure. Keep the meaning but improve style.

Draft:
{draft}"}
    ]
    improved_1 = call_chat_model(system, p1_msg, temperature=0.25, max_tokens=1800)

    # Pass 2 — humanization
    p2_msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Rewrite the improved draft to sound more human, engaging, and storytelling-driven. No fluff.

Draft:
{improved_1}"}
    ]
    improved_2 = call_chat_model(system, p2_msg, temperature=0.35, max_tokens=1800)

    # Pass 3 — SEO polish
    p3_msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Optimize the final draft for SEO while keeping it natural. Add subtle keywords. Provide:
1 paragraph summary
3 SEO titles
1 meta description

Draft:
{improved_2}"}
    ]
    final_result = call_chat_model(system, p3_msg, temperature=0.2, max_tokens=1800)

    return final_result


def generate_seo_metadata(topic: str, draft: str) -> Dict:
    system = "You are Locomind, a savvy SEO copywriter."
    user = (
        f"Given the topic: {topic} and the draft article below, produce:\n"
        "1) Three SEO-optimized titles (<= 70 chars).\n2) One meta description (~140-160 chars).\n3) Five keyword phrases (comma separated).\n4) Suggested slug (URL friendly).\n\nDraft:\n" + draft
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    txt = call_chat_model(system, messages, temperature=0.25, max_tokens=400)
    # Simple attempt to parse structure; return raw as fallback
    return {"raw": txt}


def save_article(title: str, body: str, metadata: Dict, out_dir: str = "./outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    safe_title = "_".join(title.lower().split())[:80]
    filepath = os.path.join(out_dir, safe_title + ".md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if metadata.get("meta"):
            f.write(f"<!-- meta: {metadata.get('meta')} -->\n\n")
        f.write(body)
    return filepath


# ------------------------- Orchestration -------------------------

def run_locomind(topic: str, tone: str = "neutral", length: str = "medium", use_wiki: bool = True) -> Dict:
    print(f"[Locomind] Starting pipeline for topic: {topic}")

    # 1. Understand topic
    brief = understand_topic(topic, tone=tone)["brief"]
    print("[Locomind] Brief created.")

    # 2. Web research (optional)
    research_text = ""
    if use_wiki and WIKI_AVAILABLE:
        wiki_snip = web_research_wikipedia(topic, sentences=6)
        if wiki_snip:
            research_text += wiki_snip + "\n\n"
            print("[Locomind] Wikipedia research added.")

    # Append model-sourced quick research
    research_prompt = (
        f"Provide 6 concise, fact-like bullets about '{topic}' useful for writing a blog. "
        "Label sources if known."
    )
    research_messages = [
        {"role": "system", "content": "You are Locomind, a research assistant."},
        {"role": "user", "content": research_prompt}
    ]
    model_research = call_chat_model("", research_messages, temperature=0.2, max_tokens=400)
    research_text += model_research
    print("[Locomind] Model research generated.")

    # 3. Outline
    outline = build_outline(topic, brief, tone=tone)["outline_text"]
    print("[Locomind] Outline created.")

    # 4. Draft
    draft = generate_draft(topic, outline, research_text, tone=tone, length=length)
    print("[Locomind] Draft generated.")

    # 5. Refine
    refined = refine_draft(draft)
    print("[Locomind] Draft refined.")

    # 6. SEO metadata
    seo = generate_seo_metadata(topic, refined)
    print("[Locomind] SEO metadata suggested.")

    # Optional: try to extract a title from SEO output (naive)
    title = f"{topic.title()} - by Locomind"

    # Save final
    out_path = save_article(title, refined, {"meta": seo.get("raw")})
    print(f"[Locomind] Article saved to: {out_path}")

    return {"title": title, "path": out_path, "seo": seo, "draft": draft, "refined": refined}


# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run Locomind - Blog Writing Agent")
    p.add_argument("--topic", type=str, required=True, help="Topic/title for the blog post")
    p.add_argument("--tone", type=str, default="conversational", help="Tone: conversational / formal / neutral / technical")
    p.add_argument("--length", type=str, default="medium", choices=["short", "medium", "long"], help="Approx article length")
    p.add_argument("--no-wiki", dest="use_wiki", action="store_false", help="Disable wikipedia research fallback")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _ = run_locomind(args.topic, tone=args.tone, length=args.length, use_wiki=args.use_wiki)
    print("\n[Locomind] Pipeline complete. Review the outputs directory for results.")
