import os
import json
import uuid
import datetime
from flask import Flask, request, jsonify, render_template, abort
from dotenv import load_dotenv
import requests
from openai import OpenAI

# Optional token counter using tiktoken for accurate truncation when available
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")  # optional

# Runtime mode: 'live' (calls OpenAI) or 'dev' (calls a dev server that mimics OpenAI)
MODE = os.getenv('INFITUM_ENV', 'live')
DEV_SERVER = os.getenv('INFITUM_DEV_SERVER', '')
DEV_MODEL = os.getenv('INFITUM_DEV_MODEL', 'casperhansen/llama-3.2-3b-instruct-awq')
DEV_MAX_PROMPT_CHARS = int(os.getenv('INFITUM_DEV_MAX_PROMPT_CHARS', '1200'))
DEV_MAX_PROMPT_TOKENS = int(os.getenv('INFITUM_DEV_MAX_PROMPT_TOKENS', '0'))

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

SYSTEM_PROMPT = (
    "You are an expert curriculum designer and information architect. "
    "Adopt a Conceptual & Intuition-First teaching style: start with a one-sentence core idea, "
    "offer a short visual analogy or conceptual hook for major sections, and suggest a tiny active task or teach-back prompt where useful. "
    "Given a Topic and constraints, produce a clean, hierarchical table of contents. "
    "Focus on clarity, progressive depth, and coverage of the domain. Avoid fluff, "
    "keep titles concise, and add 1-sentence descriptions only when valuable."
)

# Keep both standard and intuitive system prompts so callers can pick a learning mode.
STANDARD_SYSTEM_PROMPT = (
    "You are an expert curriculum designer and information architect. "
    "Given a Topic and constraints, produce a clean, hierarchical table of contents. "
    "Focus on clarity, progressive depth, and coverage of the domain. Avoid fluff, "
    "keep titles concise, and add 1-sentence descriptions only when valuable."
)

INTUITIVE_SYSTEM_PROMPT = SYSTEM_PROMPT


def _get_learning_mode_from(body=None) -> str:
    """Return 'standard' or 'intuitive' based on request body override or ENV INFITUM_LEARNING_MODE.
    Priority: body['learningMode'] (if present) -> ENV -> default 'standard'."""
    lm = os.getenv('INFITUM_LEARNING_MODE', 'standard') or 'standard'
    try:
        lm = lm.strip().lower()
    except Exception:
        lm = 'standard'
    if body and isinstance(body, dict):
        for key in ('learningMode', 'learning_mode', 'learningmode', 'mode'):
            if key in body and isinstance(body.get(key), str):
                val = body.get(key).strip().lower()
                if val in ('intuitive', 'int'):
                    return 'intuitive'
                if val in ('hyper', 'hyperlearner'):
                    return 'hyper'
                return 'standard'
    return 'intuitive' if lm == 'intuitive' else 'standard'


def _choose(standard_text: str, intuitive_text: str, body=None, hyper_text: str = None) -> str:
    """Pick the prompt text according to the learning mode.

    If hyper_text is provided and the mode is 'hyper', return hyper_text.
    Otherwise return intuitive or standard accordingly.
    """
    mode = _get_learning_mode_from(body)
    if mode == 'hyper' and hyper_text is not None:
        return hyper_text
    if mode == 'intuitive':
        return intuitive_text
    return standard_text
# Hyper Learner System Prompt — harmonized structure for TOC, layered pedagogy for READ/DEEPDIVE
HYPER_SYSTEM_PROMPT = (
    "You are an expert curriculum designer who teaches using the Intuitive Systems Learning Framework. "
    "For any Table of Contents (TOC) generation, maintain the same clarity, balance, and hierarchical structure "
    "as the Intuitive (Conceptual & Intuition-First) mode — clean outline, progressive depth, and concise titles. "
    "Do NOT include long narrative sections or seven-stage breakdowns inside TOC nodes. "
    "Each section title should reflect meaningful conceptual progression (Foundations → Applications → Advanced ideas). "
    "Add short one-sentence descriptions or analogical hints only where helpful for clarity. "
    "Keep the TOC purely structural, not expository. "
    "\n\n"
    "However, when producing READ or DEEPDIVE content, fully apply the 7-stage Intuitive Systems Learning Framework: "
    "1) Foundations Mapping — prerequisite ideas and why they matter; "
    "2) Historical Context & Motivation — origins and problems solved; "
    "3) Conceptual Overview (Top-Down Intuition) — big-picture core idea and visual analogies; "
    "4) Analytical Structure (Bottom-Up Logic) — detailed mechanics, algorithms, or proofs tied to intuition; "
    "5) Quizzify — 3–6 conceptual ‘what-if’ reasoning prompts with expected outcomes; "
    "6) Integration & Application — links to related fields, teach-back prompts, and small applied tasks; "
    "7) Future Directions — next steps, research areas, and advanced extensions. "
    "\n\n"
    "Maintain an exploratory, visual tone — intuition first, structure second, reasoning last. "
    "Encourage reflection through teach-back and micro-exercises where appropriate."
)


def _slug(s: str) -> str:
    import re
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s/\-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s

def _ensure_ids(nodes, seen=None, prefix=""):
    """Ensure each node has a unique, URL-safe id and recurse children."""
    if seen is None:
        seen = set()
    out = []
    for n in (nodes or []):
        nid = str(n.get("id") or _slug(f"{prefix}/{n.get('title','')}").strip("/"))
        if nid in seen:
            nid = f"{nid}-{uuid.uuid4().hex[:4]}"
        n["id"] = nid
        seen.add(nid)
        if "children" in n and isinstance(n["children"], list):
            n["children"] = _ensure_ids(n["children"], seen, n["id"])
        out.append(n)
    return out


# Helpers for token counting and model context for dev servers
def _get_dev_model_context(model_name: str) -> int:
    """Return estimated model context window for a given dev model name.
    Can be overridden with ENV INFITUM_DEV_MODEL_CONTEXT (int).
    """
    try:
        env_default = int(os.getenv('INFITUM_DEV_MODEL_CONTEXT', '4096'))
    except Exception:
        env_default = 4096

    # Known model defaults (conservative estimates). Add entries as needed.
    mapping = {
        # Conservative default for the casperhansen Llama 3.x instruct family
        'casperhansen/llama-3.2-3b-instruct-awq': 4096,
        'stabilityai/stablelm-zephyr-3b': 4096,
    }
    return mapping.get(model_name, env_default)


def _count_tokens(text: str) -> int:
    """Count approximate tokens in text. Prefer tiktoken when available.
    Falls back to a char-based heuristic (1 token ~= 4 chars).
    """
    if not text:
        return 0
    if _HAS_TIKTOKEN:
        try:
            enc = tiktoken.get_encoding('cl100k_base')
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback heuristic: 1 token per ~4 characters
    return max(1, len(text) // 4)


def _tokens_in_messages(messages) -> int:
    """Estimate total tokens used by a list of chat messages.
    Add a small per-message overhead for role/format tokens.
    """
    if not messages:
        return 0
    total = 0
    for m in messages:
        content = (m or {}).get('content', '') or ''
        total += _count_tokens(content)
        # overhead per message for role/name/formatting
        total += 4
    return total

def _chat_json(model, temperature, system, user):
    """Call OpenAI chat.completions expecting JSON content."""
    # Choose target URL depending on MODE
    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    if MODE == 'dev':
        if not DEV_SERVER:
            abort(500, "DEV server URL not configured (pass --server when running in dev mode)")
        # Talk to the dev server using the Chat Completions endpoint and send chat-style messages.
        url = DEV_SERVER.rstrip('/') + '/v1/chat/completions'

        # Helper: char-based middle truncation
        def _truncate_middle_chars(s, max_chars):
            if not s or len(s) <= max_chars:
                return s
            half = max_chars // 2
            return s[:half] + "\n\n...[truncated context]...\n\n" + s[-half:]

        # Helper: token-aware truncation using tiktoken when available
        def _truncate_by_tokens(s, max_tokens):
            if not s or max_tokens <= 0:
                return s
            if _HAS_TIKTOKEN:
                try:
                    enc = tiktoken.get_encoding('cl100k_base')
                    token_ids = enc.encode(s)
                    if len(token_ids) <= max_tokens:
                        return s
                    head = max_tokens // 2
                    tail = max_tokens - head
                    head_dec = enc.decode(token_ids[:head])
                    tail_dec = enc.decode(token_ids[-tail:])
                    return head_dec + "\n\n...[truncated context]...\n\n" + tail_dec
                except Exception:
                    pass
            # Fallback approximate by words
            words = s.split()
            if len(words) <= max_tokens:
                return s
            head = max_tokens // 2
            tail = max_tokens - head
            return ' '.join(words[:head]) + "\n\n...[truncated context]...\n\n" + ' '.join(words[-tail:])

        # Per-message truncation: prefer token-based when configured, else char-based
        def _truncate_message(s: str) -> str:
            if not s:
                return s or ""
            if DEV_MAX_PROMPT_TOKENS and isinstance(DEV_MAX_PROMPT_TOKENS, int) and DEV_MAX_PROMPT_TOKENS > 0:
                truncated = _truncate_by_tokens(s, DEV_MAX_PROMPT_TOKENS)
                if truncated != s:
                    print(f"Truncated a dev message to token limit {DEV_MAX_PROMPT_TOKENS}")
                return truncated
            if DEV_MAX_PROMPT_CHARS and isinstance(DEV_MAX_PROMPT_CHARS, int) and DEV_MAX_PROMPT_CHARS > 0:
                if len(s) > DEV_MAX_PROMPT_CHARS:
                    print(f"Truncating a dev message from {len(s)} to {DEV_MAX_PROMPT_CHARS} chars")
                    return _truncate_middle_chars(s, DEV_MAX_PROMPT_CHARS)
            return s

        # Build chat-style messages and truncate each message individually for small dev models
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": _truncate_message(system)})
        if user:
            chat_messages.append({"role": "user", "content": _truncate_message(user)})

        model_to_use = DEV_MODEL if DEV_MODEL else model
        # Determine safe max_tokens based on model context and input tokens
        requested_max = 800
        model_context = _get_dev_model_context(model_to_use)
        input_tokens = _tokens_in_messages(chat_messages)
        safe_max = max(0, min(requested_max, model_context - input_tokens - 10))
        payload = {
            "model": model_to_use,
            "messages": chat_messages,
            "temperature": temperature,
            # dynamically compute max_tokens so it fits within model context (leave a 10-token buffer)
            "max_tokens": 650
        }
        # Debug logging to help diagnose 404/Not Found from dev servers
        try:
            print(f"DEV -> POST {url}")
            print("DEV -> payload:", (payload if len(str(payload)) < 2000 else str(payload)[:2000] + '...'))
            print("DEV -> headers:", {k: ('<REDACTED>' if k.lower() == 'authorization' else v) for k, v in headers.items()})
        except Exception:
            pass

        def do_post(p):
            rr = requests.post(url, headers=headers, json=p, timeout=60)
            return rr

        r = do_post(payload)

        # If model-not-found (404 from some dev servers), try fallback DEV_MODEL if different
        if r.status_code == 404 and DEV_MODEL and payload.get('model') != DEV_MODEL:
            print(f"Dev server reported 404 for model {payload.get('model')}, retrying with {DEV_MODEL}")
            payload['model'] = DEV_MODEL
            r = do_post(payload)

        if not r.ok:
            print(f"DEV server returned status {r.status_code}")
            try:
                print("DEV response body:", r.text)
            except Exception:
                pass
            abort(r.status_code, r.text)

        data = r.json()
        print("DEV response data:", data)
        # Accept multiple shapes: choices[0].message.content or choices[0].text
        choice0 = data.get('choices', [{}])[0] or {}
        text = (choice0.get('message') or {}).get('content') or choice0.get('text') or ''

        # Try to parse JSON from the returned text. Handle common dev-server formats:
        # - Raw JSON
        # - JSON inside a ```json ... ``` fenced code block
        # - JSON embedded somewhere in the text (first {...} match)
        import re
        txt = (text or "").strip()

        # Direct parse
        try:
            parsed = json.loads(txt)
            return parsed, data
        except Exception:
            pass

        # Fenced ```json ... ``` blocks
        m = re.search(r"```json\s*([\s\S]*?)```", txt, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            try:
                parsed = json.loads(candidate)
                return parsed, data
            except Exception:
                pass

        # First JSON object substring (non-greedy)
        m2 = re.search(r"\{[\s\S]*?\}", txt)
        if m2:
            candidate = m2.group(0)
            try:
                parsed = json.loads(candidate)
                return parsed, data
            except Exception:
                pass

        # As a last attempt, try parsing choice.message.content directly
        try:
            alt = (choice0.get('message') or {}).get('content', '')
            parsed = json.loads(alt or "{}")
            return parsed, data
        except Exception:
            abort(500, f"Dev server returned invalid JSON in completions text; raw text starts: {txt[:400]}")
    else:
        # Live OpenAI Chat completions path (chat messages + JSON response format)
        url = 'https://api.openai.com/v1/chat/completions'
        payload = {
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if not r.ok:
            abort(r.status_code, r.text)
        data = r.json()
        content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or "{}"
        return json.loads(content), data

def _chat_text(model, temperature, system, messages):
    """Call OpenAI chat.completions expecting text output."""
    # Choose target URL depending on MODE
    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    if MODE == 'dev':
        if not DEV_SERVER:
            abort(500, "DEV server URL not configured (pass --server when running in dev mode)")
        # Use chat completions endpoint and send messages rather than a single prompt
        url = DEV_SERVER.rstrip('/') + '/v1/chat/completions'

        # Helpers shared with _chat_json behavior: char/token truncation and per-message truncation
        def _truncate_middle_chars(s, max_chars):
            if not s or len(s) <= max_chars:
                return s
            half = max_chars // 2
            return s[:half] + "\n\n...[truncated context]...\n\n" + s[-half:]

        def _truncate_by_tokens(s, max_tokens):
            if not s or max_tokens <= 0:
                return s
            if _HAS_TIKTOKEN:
                try:
                    enc = tiktoken.get_encoding('cl100k_base')
                    token_ids = enc.encode(s)
                    if len(token_ids) <= max_tokens:
                        return s
                    head = max_tokens // 2
                    tail = max_tokens - head
                    head_dec = enc.decode(token_ids[:head])
                    tail_dec = enc.decode(token_ids[-tail:])
                    return head_dec + "\n\n...[truncated context]...\n\n" + tail_dec
                except Exception:
                    pass
            words = s.split()
            if len(words) <= max_tokens:
                return s
            head = max_tokens // 2
            tail = max_tokens - head
            return ' '.join(words[:head]) + "\n\n...[truncated context]...\n\n" + ' '.join(words[-tail:])

        def _truncate_message(s: str) -> str:
            if not s:
                return s or ""
            if DEV_MAX_PROMPT_TOKENS and isinstance(DEV_MAX_PROMPT_TOKENS, int) and DEV_MAX_PROMPT_TOKENS > 0:
                truncated = _truncate_by_tokens(s, DEV_MAX_PROMPT_TOKENS)
                if truncated != s:
                    print(f"Truncated a dev message to token limit {DEV_MAX_PROMPT_TOKENS}")
                return truncated
            if DEV_MAX_PROMPT_CHARS and isinstance(DEV_MAX_PROMPT_CHARS, int) and DEV_MAX_PROMPT_CHARS > 0:
                if len(s) > DEV_MAX_PROMPT_CHARS:
                    print(f"Truncating a dev message from {len(s)} to {DEV_MAX_PROMPT_CHARS} chars")
                    return _truncate_middle_chars(s, DEV_MAX_PROMPT_CHARS)
            return s

        # Build chat messages and truncate each individually
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": _truncate_message(system)})
        for m in messages:
            chat_messages.append({"role": m.get("role", "user"), "content": _truncate_message(m.get("content", ""))})

        model_to_use = DEV_MODEL if DEV_MODEL else model
        # Determine safe max_tokens based on model context and input tokens
        requested_max = 800
        model_context = _get_dev_model_context(model_to_use)
        input_tokens = _tokens_in_messages(chat_messages)
        safe_max = max(0, min(requested_max, model_context - input_tokens - 10))
        payload = {
            "model": model_to_use,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": 650
        }
        try:
            print(f"DEV -> POST {url}")
            print("DEV -> payload:", (payload if len(str(payload)) < 2000 else str(payload)[:2000] + '...'))
            print("DEV -> headers:", {k: ('<REDACTED>' if k.lower() == 'authorization' else v) for k, v in headers.items()})
        except Exception:
            pass

        def do_post(p):
            return requests.post(url, headers=headers, json=p, timeout=60)

        r = do_post(payload)
        # retry with DEV_MODEL if model-not-found
        if r.status_code == 404 and DEV_MODEL and payload.get('model') != DEV_MODEL:
            print(f"Dev server reported 404 for model {payload.get('model')}, retrying with {DEV_MODEL}")
            payload['model'] = DEV_MODEL
            r = do_post(payload)

        if not r.ok:
            print(f"DEV server returned status {r.status_code}")
            try:
                print("DEV response body:", r.text)
            except Exception:
                pass
            abort(r.status_code, r.text)

        data = r.json()
        choice0 = data.get('choices', [{}])[0] or {}
        text = (choice0.get('message') or {}).get('content') or choice0.get('text') or ''
        return text, data
    else:
        url = 'https://api.openai.com/v1/chat/completions'
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}] + messages,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if not r.ok:
            abort(r.status_code, r.text)
        data = r.json()
        content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
        return content, data

@app.route("/")
def index():
    # Renders your existing HTML/JS, lightly edited to call this Flask API.
    return render_template("index.html", youtube_enabled=bool(YOUTUBE_API_KEY))


@app.route('/notebook/<path:name>')
def notebook_route(name):
    # Serve the same SPA for notebook paths so the frontend can read the path and load the requested notebook.
    return render_template("index.html", youtube_enabled=bool(YOUTUBE_API_KEY))

@app.post("/api/toc")
def api_toc():
    body = request.get_json(force=True, silent=True) or {}
    topic = body.get("topic", "").strip()
    if not topic:
        abort(400, "Topic is required.")
    audience = body.get("audience", "general")
    depth = int(body.get("depth", 3))
    sections = int(body.get("sections", 5))
    model = body.get("model", "gpt-4.1-mini")
    temperature = float(body.get("temperature", 0.3))

    intuitive_user_prompt = f"""
Topic: {topic}
Audience: {audience}
Desired depth/levels: {depth}
Target sections per level (approx): {sections}

Pedagogy: Use a Conceptual & Intuition-First style where possible: start with a one-sentence core idea for the topic, prefer short visual analogies or hooks for top-level sections, and suggest a tiny teach-back prompt or 1-line exercise when helpful.

Requirements:
- Return ONLY valid JSON matching the schema {{ toc: TocNode[] }}.
- Each TocNode: {{ id: string (slug), title: string, description?: string, children?: TocNode[] }}
- IDs must be unique, URL-safe slugs derived from titles (e.g., "neural-networks/activation-functions").
- The tree should be reasonably balanced and non-redundant.
- Include foundational -> intermediate -> advanced progression.
- Prefer 3–7 top-level sections unless topic is very narrow.
""".strip()

    standard_user_prompt = f"""
Topic: {topic}
Audience: {audience}
Desired depth/levels: {depth}
Target sections per level (approx): {sections}

Requirements:
- Return ONLY valid JSON matching the schema {{ toc: TocNode[] }}.
- Each TocNode: {{ id: string (slug), title: string, description?: string, children?: TocNode[] }}
- IDs must be unique, URL-safe slugs derived from titles (e.g., "neural-networks/activation-functions").
- The tree should be reasonably balanced and non-redundant.
- Include foundational -> intermediate -> advanced progression.
- Prefer 3–7 top-level sections unless topic is very narrow.
""".strip()

    # Hyper Learner user prompt (follows the Intuitive Systems Learning Framework)
    hyper_user_prompt = f"""
Topic: {topic}
Audience: {audience}
Desired depth/levels: {depth}
Target sections per level (approx): {sections}

Instructions (Hyper Learner): Follow the Intuitive Systems Learning Framework stages.
- Foundations Mapping: list essential prerequisites (4–8) and a 1–2 sentence note on why each prerequisite is required.
- Historical Context & Motivation: give a concise origin story explaining the problems this topic was created to solve.
- Conceptual Overview: supply a top-down mental model, visual analogy(s), and a short 'core idea' sentence.
- Analytical Structure: outline the main components, mechanisms, or algorithms and show how they connect to the top-down model.
- Quizzify: give 3–6 conceptual 'what-if' or reasoning questions with brief expected outcomes or hints.
- Integration & Application: map links to 3 related topics and suggest 1–2 small applied exercises or teach-back prompts.
- Future Directions: list 3 advanced or frontier directions to explore next.

Requirements:
- Return ONLY valid JSON matching the schema {{ toc: TocNode[] }}.
- Each TocNode: {{ id: string (slug), title: string, description?: string, children?: TocNode[] }}
- Keep top-level sections meaningful, include short analogies or 'core idea' snippets for high-level nodes when helpful.
""".strip()

    user_prompt = _choose(standard_user_prompt, intuitive_user_prompt, body, hyper_text=hyper_user_prompt)
    system_choice = _choose(STANDARD_SYSTEM_PROMPT, INTUITIVE_SYSTEM_PROMPT, body, hyper_text=HYPER_SYSTEM_PROMPT)
    parsed, raw = _chat_json(model, temperature, system_choice, user_prompt)
    toc = _ensure_ids(parsed.get("toc", []))

    # Best-effort: also fetch content for the first top-level node so the client
    # can render the first chapter immediately without an extra round-trip.
    first_content = None
    first_content_meta = None
    try:
        if toc and len(toc) > 0:
            first = toc[0]
            # Build a lightweight read prompt similar to /api/read (defaults to Beginner / level 1)
            level = 1
            level_descriptions = {
                1: "Explain for a beginner: simple analogies, everyday examples, avoid jargon."
            }
            level_instruction = level_descriptions.get(level, "")

            user_prompt_read = f"""
Write a clear, well-structured explanation (250–450 words) for the selected outline item. Use Markdown and LaTeX for math.

Global Topic: {topic}
Audience: {audience}
Proficiency Level: {level}/10 - {level_instruction}
Depth preference (context): {depth}
Approx sections per level (context): {sections}

Breadcrumb (root->current): {''}
Current Item: {first.get('title','')}
Current Item ID: {first.get('id','')}

Style:
- Use paragraphs and short bullet lists where helpful.
- Inline math: $...$ ; display math: $$...$$
- Match the requested proficiency level exactly: {level_instruction}
- No JSON.
""".strip()

            content, raw2 = _chat_text(model, temperature,
                                      "You explain concepts clearly with examples. Use Markdown and LaTeX for math.",
                                      [{"role": "user", "content": user_prompt_read}])
            first_content = content
            first_content_meta = {"model": raw2.get("model"), "tokens": raw2.get("usage")}
    except Exception as e:
        # Do not fail the entire TOC generation if the read call fails; log and continue.
        print(f"Warning: fetching first node content with /api/toc failed: {e}")

    resp_obj = {"toc": toc, "model": raw.get("model"), "tokens": raw.get("usage")}
    if first_content is not None:
        resp_obj["firstContent"] = first_content
        if first_content_meta:
            resp_obj["firstContentMeta"] = first_content_meta
    return jsonify(resp_obj)


@app.post("/api/foundations")
def api_foundations():
    body = request.get_json(force=True, silent=True) or {}
    topic = body.get("topic", "").strip()
    if not topic:
        abort(400, "Topic is required.")
    audience = body.get("audience", "general")
    depth = int(body.get("depth", 3))
    sections = int(body.get("sections", 5))
    model = body.get("model", "gpt-4.1-mini")
    temperature = float(body.get("temperature", 0.3))

    # Optional node context (id/title) to focus the Foundations guide
    node = body.get("node") or {}
    first = node or {}

    level = 1
    level_instruction = "Explain for a beginner: simple analogies, everyday examples, avoid jargon."
    standard_user_prompt_read = f"""
Write an exhaustive, structured "Foundations" guide that lists the *necessary prerequisite topics* a learner must master to thoroughly understand the selected outline item. Produce the output in Markdown (headings, short paragraphs, and bullet lists) suitable as a study checklist and teaching scaffold.

Global Topic: {topic}
Audience: {audience}
Proficiency Level: {level}/10 - {level_instruction}
Depth preference (context): {depth}
Approx sections per level (context): {sections}

Breadcrumb (root->current): {''}
Current Item: {first.get('title','')}
Current Item ID: {first.get('id','')}

Requirements & structure:
- Length: ~400–900 words (aim for thoroughness and clarity).
- Output ONLY Markdown (no JSON).
- Top-level heading: "# Foundations for <Current Item>" (use the actual current item title).
- Start with a 2–3 sentence summary describing the role of these foundations — how the prerequisites enable understanding of the current item.
- Provide an ordered list of 4–8 **Essential Prerequisite Topics**. For *each* prerequisite topic include:
  1. A clear subheading (###) with a concise title (3–6 words).
  2. A focused overview (2–4 sentences) explaining *what* the prerequisite is and *why* it matters specifically for the current item.
  3. A "Key pointers" bullet list (3–5 actionable items), including:
       - concrete concepts to master,
       - minimal formulas or definitions to memorize,
       - canonical examples to study,
       - short practice tasks or exercises, and
       - one concise recommended resource (textbook chapter title, keyword, or short URL phrase).
- After the prerequisite list include:
  - "Minimal skills checklist" (3–8 short bullets of concrete abilities a learner should have before proceeding).
  - "Suggested study order" (2–6 sequential steps to learn the prerequisites efficiently).
  - "Concise resource recommendations" (1–4 short entries—titles or brief links—to begin learning).
- Use $...$ for inline math and $$...$$ for display math where a formula is necessary. Do not put math inside code fences.
- Tone: instructive, practical, audience-appropriate; match the specified proficiency level and avoid abstract platitudes.

Style guidelines:
- Prioritize concreteness: each prerequisite overview should tell the learner exactly *what* to study and *how* to practice it.
- Use headings, short paragraphs, and bullet lists for scan-ability.
- Avoid lengthy historical narrative unless it directly explains why a prerequisite exists; prefer actionable learning advice.
- Do not produce JSON or extra metadata.

Example outline (format only):
# Foundations for {first.get('title','')}
Short summary...
""".strip()

    intuitive_user_prompt_read = standard_user_prompt_read + "\n\nPedagogy guidance:\n- Use a Conceptual & Intuition-First style: start the guide with one-sentence core idea and a short visual analogy that links the prerequisites to the current item.\n- For each prerequisite, where possible, include a 1-line \"teach-back\" prompt (e.g., \"Explain X in one sentence\" or \"Draw a quick sketch showing Y\") the learner can use to test understanding.\n- Prefer concrete examples and 1–2 micro-exercises that build intuition rather than rote memorization.\n"

    # Hyper Learner variant for Foundations/read: follow the 7-stage Intuitive Systems Learning Framework
    hyper_user_prompt_read = standard_user_prompt_read + "\n\nHyper Learner guidance:\n- Start with Foundations Mapping: list prerequisites and why they matter.\n- Provide Historical Context: brief origin / motivation.\n- Provide a concise Conceptual Overview (visual analogy + core idea).\n- Then the Analytical Structure with stepwise components.\n- Include a Quizzify section (3-6 conceptual what-if questions with expected outcomes or hints).\n- Add Integration & Application suggestions (teach-back prompts or small applied tasks).\n- End with Future Directions and advanced next steps.\n"

    user_prompt_read = _choose(standard_user_prompt_read, intuitive_user_prompt_read, body, hyper_text=hyper_user_prompt_read)

    try:
        system_choice = _choose("You are an expert curriculum designer. Produce a Foundations guide in Markdown as specified.",
                               "You are an expert curriculum designer. Produce a Foundations guide in Markdown as specified.",
                               body, hyper_text=HYPER_SYSTEM_PROMPT)
        content, raw = _chat_text(model, temperature, system_choice, [{"role": "user", "content": user_prompt_read}])
        if not content or not isinstance(content, str):
            abort(500, "Foundations generation failed: no content returned")
        return jsonify({"content": content, "model": raw.get("model"), "tokens": raw.get("usage")})
    except Exception as e:
        abort(500, f"Foundations generation failed: {e}")

@app.post("/api/expand")
def api_expand():
    body = request.get_json(force=True, silent=True) or {}
    topic = body.get("topic", "").strip()
    node = body.get("node", {})  # expects { id, title }
    path = body.get("path", [])  # [{title,id}, ...]
    audience = body.get("audience", "general")
    depth = int(body.get("depth", 3))
    sections = int(body.get("sections", 5))
    model = body.get("model", "gpt-4.1-mini")
    temperature = float(body.get("temperature", 0.3))
    if not topic or not node:
        abort(400, "Missing topic or node.")

    standard_user_prompt = f"""
You are expanding a selected node inside a topic outline.

Global Topic: {topic}
Audience: {audience}
Desired depth/levels: {depth}
Target sections per level (approx): {sections}

Breadcrumb (root->current): {' > '.join([p.get('title','') for p in path])}
Current Node Title: {node.get('title','')}
Current Node ID: {node.get('id','')}

Return ONLY valid JSON:
{{ "children": TocNode[] }}
TocNode = {{ id: string (slug), title: string, description?: string, children?: TocNode[] }}
- Provide 3–7 high-quality sub-sections.
- IDs must be unique and URL-safe; prefix with the current node id.
- Keep titles concise; add short descriptions only when valuable.
""".strip()

    intuitive_user_prompt = standard_user_prompt + "\n\nPedagogy guidance:\n- Use a Conceptual & Intuition-First approach: aim to include a one-sentence core idea or analogy for each sub-section where helpful.\n- For at least one child, include a short \"micro-exercise\" (1–2 sentence teach-back or quick mental experiment) that solidifies intuition.\n"

    user_prompt = _choose(standard_user_prompt, intuitive_user_prompt, body)
    parsed, raw = _chat_json(model, temperature, _choose(STANDARD_SYSTEM_PROMPT, INTUITIVE_SYSTEM_PROMPT, body), user_prompt)
    children = _ensure_ids(parsed.get("children", []), prefix=node.get("id", ""))
    return jsonify({"children": children, "model": raw.get("model"), "tokens": raw.get("usage")})

@app.post("/api/read")
def api_read():
    body = request.get_json(force=True, silent=True) or {}
    topic = body.get("topic", "").strip()
    node = body.get("node", {})
    path = body.get("path", [])
    audience = body.get("audience", "general")
    depth = int(body.get("depth", 3))
    sections = int(body.get("sections", 5))
    model = body.get("model", "gpt-4.1-mini")
    temperature = float(body.get("temperature", 0.3))
    level = int(body.get("level", 1))  # Default to level 1 (Beginner) if not specified
    if not topic or not node:
        abort(400, "Missing topic or node.")

    # Level-aware prompting
    level_descriptions = {
        1: "Explain as if to a curious child or complete newcomer: use simple analogies, everyday examples, avoid jargon, no complex formulas unless absolutely necessary.",
        2: "Explain for someone with basic knowledge: use simple concepts and examples, minimal technical terms, focus on understanding over precision.",
        3: "Explain for an introductory learner: use key concepts and intuition, some examples, basic terminology, avoid advanced mathematics.",
        4: "Explain for an intermediate learner: balanced explanation with some technical details, examples and intuition, moderate use of terminology.",
        5: "Explain for an informed learner: comprehensive overview with good balance of intuition and technical details, standard terminology.",
        6: "Explain for an advanced learner: detailed explanation with mathematical concepts, technical terminology, some derivations.",
        7: "Explain for an expert: rigorous treatment with derivations, formal definitions, advanced mathematics, technical precision.",
        8: "Explain for a specialist: formal treatment with proofs, advanced formalism, specialized terminology, research-level concepts.",
        9: "Explain for a researcher: cutting-edge concepts, advanced formalism, open problems, research-level depth.",
        10: "Explain at maximum expertise level: complete formalism, rigorous proofs, advanced mathematics, research-level precision."
    }
    
    level_instruction = level_descriptions.get(level, level_descriptions[1])

    # Replace the per-case prompts with a single exhaustive-study prompt for read requests.
    standard_user_prompt = f"""
Write an exhaustive, in-depth study of the selected outline item. Treat it as a comprehensive mini-chapter that explores the topic from all relevant angles — theory, context, methodology, examples, and implications. The goal is to provide a complete understanding suitable for independent study or teaching material.

Global Topic: {topic}
Audience: {audience}
Proficiency Level: {level}/10 - {level_instruction}
Depth preference (context): {depth}
Approx sections per level (context): {sections}

Breadcrumb (root->current): {' > '.join([p.get('title','') for p in path])}
Current Item: {node.get('title','')}
Current Item ID: {node.get('id','')}

Guidelines:
- Length: 900–1500 words (aim for depth and completeness, not brevity).
- Organize clearly with Markdown subheadings (e.g., Introduction, Conceptual Foundations, Mechanisms, Mathematical Formulation, Applications, Examples, Limitations, Future Directions).
- Include formal definitions, derivations, or step-by-step reasoning where relevant.
- Provide multiple illustrative examples (mathematical, conceptual, or real-world) with brief explanations.
- Integrate diagrams or visual analogies when appropriate (describe them textually if visuals cannot be rendered).
- Use $...$ for inline math and $$...$$ for display math.
- Write in a teaching tone: structured, logical, and progressively deep.
- Do **not** return JSON; return only Markdown content.
""".strip()

    intuitive_user_prompt = standard_user_prompt + "\n\nPedagogy guidance:\n- Begin with a one-sentence \"core idea\" and a short visual analogy or conceptual hook that orients intuition.\n- Organize material in stepwise layers: conceptual foundations first, then mechanisms, then formal details and examples.\n- Where appropriate, include a small \"Try this\" teach-back: a 1–2 sentence prompt or micro-experiment the learner can do to test intuition (e.g., \"Predict what happens if X increases\") and a short expected outcome.\n- Favor clear diagrams/analogies and call them out explicitly in the text (textual descriptions are fine if visuals can't be rendered).\n"

    user_prompt = _choose(standard_user_prompt, intuitive_user_prompt, body)

    content, raw = _chat_text(model, temperature,
                              "You explain concepts clearly with examples. Use Markdown and LaTeX for math.",
                              [{"role": "user", "content": user_prompt}])
    # If this is an Overview node, also ask the model for a concise 2-4 sentence
    # summary server-side so the UI doesn't have to parse or synthesize one.
    summary_text = None
    try:
        if (node.get('title') or '').strip().lower() == 'overview':
            sum_prompt = f"""
Summarize the following overview in 2–4 clear sentences suitable as a short summary for a UI display. Keep it focused and non-redundant; do not add new technical details.

Overview content:
{content}

Return only the summary as plain text.
""".strip()
            sum_resp, raw_sum = _chat_text(model, 0.2,
                                           "You are an expert summarizer. Produce 2-4 clear sentences that capture the essence of the provided overview.",
                                           [{"role": "user", "content": sum_prompt}])
            summary_text = (sum_resp or "").strip()
    except Exception:
        # Best-effort only; don't fail the read call for summary generation errors
        summary_text = None

    resp = {"content": content, "model": raw.get("model"), "tokens": raw.get("usage")}
    if summary_text:
        resp["summary"] = summary_text
    return jsonify(resp)

@app.post("/api/chat")
def api_chat():
    """Handle a simple chat request. Expected JSON body:
    { topic, audience, node, path, question, context, history, model, temperature }
    The endpoint uses the provided `context` (if any) as the primary source for answers.
    """
    body = request.get_json(force=True, silent=True) or {}
    topic = body.get("topic", "").strip()
    audience = body.get("audience", "general")
    node = body.get("node") or {}
    path = body.get("path") or []
    question = (body.get("question") or "").strip()
    context = body.get("context", "") or ""
    history = body.get("history", []) or []
    model = body.get("model", "gpt-4.1-mini")
    temperature = float(body.get("temperature", 0.3))

    if not question:
        abort(400, "question is required")

    # Build a concise system instruction guiding the assistant to prefer the provided context
    system = (
        "You are a helpful assistant. Prefer a Conceptual & Intuition-First style: start answers with a one-sentence core idea, offer a short visual analogy when helpful, and include a 1-line teach-back prompt the user can use to check understanding. "
        "Use the provided context as the primary source when answering. If the answer is not present in the context, be honest and provide brief, relevant guidance. Use Markdown formatting and $...$ / $$...$$ for math when appropriate."
    )

    messages = []
    # If a long context is provided, include it as an initial system/user message
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})

    # Append recent history if any (user/assistant turns)
    for h in (history or [])[-10:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h.get("role"), "content": h.get("content", "")})

    # Finally add the user's question
    messages.append({"role": "user", "content": f"Question: {question}"})

    try:
        answer, raw = _chat_text(model, temperature, system, messages)
        return jsonify({"answer": answer, "model": raw.get("model"), "tokens": raw.get("usage")})
    except Exception as e:
        abort(500, f"Chat generation failed: {e}")

@app.get("/api/videos")
def api_videos():
    """Return 3–6 YouTube search results for ?q=... (uses server-side key if configured).
       Falls back to mock thumbnails if no key is set."""
    q = request.args.get("q", "").strip()
    if not q:
        abort(400, "q is required")
    if not YOUTUBE_API_KEY:
        # mock fallback
        base = "https://placehold.co/480x270?text="
        items = []
        for i in range(1, 7):
            items.append({
                "title": f"Learning: {q} — Part {i}",
                "channel": "Demo Channel",
                "duration": "10:0{}".format(i),
                "thumbnail": base + q[:18].replace(" ", "+"),
                "url": f"https://www.youtube.com/results?search_query={q.replace(' ', '+')}"
            })
        return jsonify({"videos": items})

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "type": "video",
        "maxResults": 6,
        "q": q,
        "key": YOUTUBE_API_KEY,
        "safeSearch": "moderate",
    }
    r = requests.get(url, params=params, timeout=20)
    if not r.ok:
        # graceful fallback
        return jsonify({"videos": [{
            "title": f"Search YouTube for {q}",
            "channel": "YouTube",
            "duration": "",
            "thumbnail": "https://placehold.co/480x270?text=YouTube",
            "url": f"https://www.youtube.com/results?search_query={q.replace(' ', '+')}"
        }]})
    data = r.json()
    videos = []
    for it in data.get("items", []):
        vid = (it.get("id") or {}).get("videoId")
        if not vid:
            continue
        title = (it.get("snippet") or {}).get("title", "Video")
        channel = (it.get("snippet") or {}).get("channelTitle", "Channel")
        thumb = (((it.get("snippet") or {}).get("thumbnails") or {}).get("high") or
                 ((it.get("snippet") or {}).get("thumbnails") or {}).get("medium") or
                 ((it.get("snippet") or {}).get("thumbnails") or {}).get("default") or {}).get("url") \
                or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
        videos.append({
            "title": title,
            "channel": channel,
            "duration": "",
            "thumbnail": thumb,
            "url": f"https://www.youtube.com/watch?v={vid}"
        })
    return jsonify({"videos": videos})

@app.post("/api/deepdive")
def api_deepdive():
    """Create child sections based on a text selection from a parent node."""
    body = request.get_json(force=True, silent=True) or {}
    topic = body.get("topic", "").strip()
    node = body.get("node", {})
    path = body.get("path", [])
    audience = body.get("audience", "general")
    model = body.get("model", "gpt-4.1-mini")
    temperature = float(body.get("temperature", 0.3))
    selection = body.get("selection", {})
    with_context = bool(body.get("with_context", False))

    if not topic or not node or not selection:
        abort(400, "Missing topic, node, or selection.")

    selection_text = selection.get("text", "").strip()
    if len(selection_text) < 8:
        abort(400, "Selection too short. Please select at least 8 characters.")
    if len(selection_text) > 3000:
        abort(400, "Selection too long. Please select less than 3000 characters.")

    if not with_context:
        # Neutral/standalone prompt: do not mention global topic or breadcrumb
        system_prompt = (
            "You are an expert curriculum designer creating focused, standalone educational content. "
            "Adopt a Conceptual & Intuition-First style: in the Overview include a one-sentence core idea, a short analogy or visual hook, and a 1-line teach-back micro-exercise. "
            "Given a short text selection, produce 1-3 child sections that explain and deepen understanding of that selection. Do NOT reference the parent topic, breadcrumb, or surrounding context."
        )

        user_prompt = f"""
You are creating a focused deep dive structure based solely on the provided text selection.

Selected Text to Deep Dive Into:
"{selection_text}"

Instructions:
- Generate a shortName (3-6 words, noun-phrase) that captures the essence of the selection
- Create children array where:
  - children[0] is "Overview" with rich readContent (200-300 words, Markdown + LaTeX) explaining the selection
  - children[1..] are 2-3 relevant subtopics with concise titles and 1-2 line descriptions (no readContent)
- Focus tightly on the selected passage; do NOT include or rely on parent/topic-level context
- Use clear, educational titles (not sentences)
- Keep IDs URL-safe

Return ONLY valid JSON:
{{ 
  "shortName": string,
  "wrapperDescription": string (optional),
  "children": TocNode[] 
}}

TocNode = {{ 
  id?: string (optional), 
  title: string, 
  description?: string, 
  readContent?: string (Markdown with LaTeX)
}}

Style guidelines:
- shortName should be a concise noun-phrase (e.g., "Activation Functions")
- Overview readContent should be comprehensive (200-300 words) with Markdown and LaTeX
- For math expressions, use $...$ for inline math or $$...$$ for display math
- Do not surround TeX with normal parentheses or brackets, and do not put math inside code fences
""".strip()

    else:
        # Context-aware prompt: include topic and breadcrumb
        system_prompt = (
            "You are an expert curriculum designer creating focused, educational content. "
            "Adopt a Conceptual & Intuition-First style: in the Overview include a one-sentence core idea, a short analogy or visual hook, and a 1-line teach-back micro-exercise. "
            "Given a text selection from a parent topic, create 1-3 child sections that meaningfully "
            "deepen understanding of the selected concept. Focus on clarity, progressive learning, "
            "and practical application."
        )

        user_prompt = f"""
You are creating a focused deep dive structure based on a specific text selection from a parent topic.

Global Topic: {topic}
Audience: {audience}
Breadcrumb (root->current): {' > '.join([p.get('title','') for p in path])}
Parent Node: {node.get('title','')} (ID: {node.get('id','')})

Selected Text to Deep Dive Into:
"{selection_text}"

Instructions:
- Generate a shortName (3-6 words, noun-phrase) that captures the essence of the selection
- Create children array where:
  - children[0] is "Overview" with rich readContent (200-300 words, Markdown + LaTeX) explaining the selection
  - children[1..] are 2-3 relevant subtopics with concise titles and 1-2 line descriptions (no readContent)
- Focus on the specific selection, not general background
- Use clear, educational titles (not sentences)
- Keep IDs URL-safe

Return ONLY valid JSON:
{{ 
  "shortName": string,
  "wrapperDescription": string (optional),
  "children": TocNode[] 
}}

TocNode = {{ 
  id?: string (optional), 
  title: string, 
  description?: string, 
  readContent?: string (Markdown with LaTeX)
}}

Style guidelines:
- shortName should be a concise noun-phrase (e.g., "Neural Network Activation Functions")
- wrapperDescription should be a one-liner explaining the focus area
- Overview title should be "Overview" or a short phrase
- Overview readContent should be comprehensive (200-300 words) with Markdown and LaTeX
- Other children should have concise titles and brief descriptions only
- Focus on the specific selection, not general background
- Maintain the audience level and tone
- For math expressions, use $...$ for inline math or $$...$$ for display math
- Do not surround TeX with normal parentheses or brackets, and do not put math inside code fences
""".strip()

    # Choose the system prompt according to learning mode (supports hyper)
    system_choice = _choose(system_prompt, system_prompt, body, hyper_text=HYPER_SYSTEM_PROMPT)
    # If hyper mode is requested, append explicit hyper guidance to the user prompt
    if _get_learning_mode_from(body) == 'hyper':
        hyper_add = "\n\nHyper Learner guidance:\n- In the Overview include Foundations Mapping (prereqs), Historical Context, Conceptual Overview with visual analogy, Analytical Structure, Quizzify (3 conceptual questions + brief expected answers), Integration exercises, and Future Directions.\n- Keep Overview ~200-300 words and Quizzify concise and conceptual.\n"
        user_prompt = user_prompt + hyper_add
    parsed, raw = _chat_json(model, temperature, system_choice, user_prompt)
    
    # Extract the response components
    short_name = parsed.get("shortName", "").strip()
    wrapper_description = parsed.get("wrapperDescription", "").strip()
    children = parsed.get("children", [])
    
    if not short_name or not children:
        abort(400, "Invalid response: missing shortName or children")
    
    # Ensure children have proper IDs and limit to 4 children max (Overview + 3 subtopics)
    children = _ensure_ids(children[:4], prefix="")
    
    # Ensure first child is Overview with readContent
    if children and children[0].get("title", "").lower() != "overview":
        children[0]["title"] = "Overview"
    
    return jsonify({
        "shortName": short_name,
        "wrapperDescription": wrapper_description,
        "children": children,
        "model": raw.get("model"), 
        "tokens": raw.get("usage")
    })


@app.post("/api/quickdive")
def api_quickdive():
    """Create a short 2-3 paragraph overview + key topics for a selected text.
       Returns plain Markdown/text in the `content` field."""
    body = request.get_json(force=True, silent=True) or {}
    topic = body.get("topic", "").strip()
    node = body.get("node", {})
    path = body.get("path", [])
    audience = body.get("audience", "general")
    model = body.get("model", "gpt-4.1-mini")
    temperature = float(body.get("temperature", 0.3))
    selection = (body.get("selection") or {}).get("text", "").strip()

    if not selection:
        abort(400, "Selection text is required for quick dive")

    # Standard variant (previous behavior) — concise quick overview without explicit teach-back/pedagogy lines
    standard_user_prompt = f"""
You are an expert explainer. Given the following short selected text, produce a concise quick overview intended for a reader familiar with the surrounding topic.

Selected text:
{selection}

Requirements:
- Provide a clear 2-3 paragraph overview (each paragraph short, easy to scan).
- After the overview, include a short "Key topics" section listing the main topics or concepts (3-6 bullet points).
- If an illustrative example helps, include a single short example at the end under an "Example" heading.
- Use Markdown formatting (paragraphs, a bullet list for Key topics, and an optional Example section).
- Keep the language accessible and focused; do not produce JSON or extra metadata.
""".strip()

    intuitive_user_prompt = f"""
You are an expert explainer. Given the following short selected text, produce a concise quick overview intended for a reader familiar with the surrounding topic.

Selected text:
{selection}

Requirements:
- Provide a clear 2-3 paragraph overview (each paragraph short, easy to scan). Start with a one-sentence core idea and a short analogy if useful.
- After the overview, include a short "Key topics" section listing the main topics or concepts (3-6 bullet points).
- Include a single 1-line teach-back prompt or micro-exercise at the end (e.g., "Explain X in one sentence" or "Predict what happens if Y doubles").
- If an illustrative example helps, include a single short example at the end under an "Example" heading.
- Use Markdown formatting (paragraphs, a bullet list for Key topics, and an optional Example section).
- Keep the language accessible and focused; do not produce JSON or extra metadata.
""".strip()

    user_prompt = _choose(standard_user_prompt, intuitive_user_prompt, body)

    try:
        content, raw = _chat_text(model, temperature,
                                 "You are a clear, concise explainer. Produce 2-3 short paragraphs, followed by a Key topics list and an optional short Example.",
                                 [{"role": "user", "content": user_prompt}])
    except Exception as e:
        abort(500, f"Quick dive generation failed: {e}")

    return jsonify({"content": content})

@app.post("/api/notebook/save")
def api_notebook_save():
    """Save a notebook to server storage (optional cloud persistence)."""
    body = request.get_json(force=True, silent=True) or {}
    NOTEBOOK_DIR = os.getenv("NOTEBOOK_DIR", "notebooks")
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)

    notebook_id = body.get("id") or str(uuid.uuid4())
    name = (body.get("name") or "untitled").strip()
    notebook_data = body.get("notebook")

    if not notebook_data:
        abort(400, "Notebook data is required.")

    # Persist to a JSON file on the server. Filename is the notebook_id.json
    path = os.path.join(NOTEBOOK_DIR, f"{notebook_id}.json")
    now = datetime.datetime.utcnow().isoformat() + "Z"

    # If file exists, preserve createdAt
    created_at = now
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                created_at = existing.get("createdAt", created_at)
        except Exception:
            created_at = now

    envelope = {
        "id": notebook_id,
        "name": name,
        "createdAt": created_at,
        "updatedAt": now,
        "notebook": notebook_data,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2)
    except Exception as e:
        abort(500, f"Failed to save notebook: {e}")

    return jsonify({
        "id": notebook_id,
        "name": name,
        "saved": True,
        "message": "Notebook saved successfully",
        "updatedAt": now,
    })

@app.get("/api/notebook/load")
def api_notebook_load():
    """Load a notebook from server storage."""
    NOTEBOOK_DIR = os.getenv("NOTEBOOK_DIR", "notebooks")
    notebook_id = request.args.get("id")
    if not notebook_id:
        abort(400, "Notebook ID is required.")

    path = os.path.join(NOTEBOOK_DIR, f"{notebook_id}.json")
    if not os.path.exists(path):
        abort(404, "Notebook not found")

    try:
        with open(path, "r", encoding="utf-8") as f:
            envelope = json.load(f)
    except Exception as e:
        abort(500, f"Failed to read notebook: {e}")

    return jsonify(envelope)


@app.get("/api/notebook/list")
def api_notebook_list():
    """List notebooks saved on the server."""
    NOTEBOOK_DIR = os.getenv("NOTEBOOK_DIR", "notebooks")
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    items = []
    for fname in os.listdir(NOTEBOOK_DIR):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(NOTEBOOK_DIR, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                envelope = json.load(f)
                items.append({
                    'id': envelope.get('id'),
                    'name': envelope.get('name'),
                    'createdAt': envelope.get('createdAt'),
                    'updatedAt': envelope.get('updatedAt'),
                })
        except Exception:
            # Skip unreadable files
            continue

    # Sort by updatedAt desc
    items.sort(key=lambda x: x.get('updatedAt') or '', reverse=True)
    return jsonify({'notebooks': items})


@app.post("/api/notebook/delete")
def api_notebook_delete():
    """Delete a notebook file by id from server storage."""
    body = request.get_json(force=True, silent=True) or {}
    notebook_id = body.get('id')
    NOTEBOOK_DIR = os.getenv("NOTEBOOK_DIR", "notebooks")
    if not notebook_id:
        abort(400, "Notebook ID is required for deletion")

    path = os.path.join(NOTEBOOK_DIR, f"{notebook_id}.json")
    if not os.path.exists(path):
        abort(404, "Notebook not found")

    try:
        os.remove(path)
    except Exception as e:
        abort(500, f"Failed to delete notebook: {e}")

    return jsonify({"id": notebook_id, "deleted": True, "message": "Notebook deleted"})

@app.route("/api/visualize/eligibility", methods=["POST"])
def visualize_eligibility():
    """Determine if a node would benefit from visualization"""
    try:
        body = request.get_json()
        node = body.get("node", {})
        content = body.get("content", "")
        topic = body.get("topic", "")
        
        if not node.get("title"):
            return jsonify({"ok": False, "score": 0, "rationale": "No node title provided"})
        
        # Create eligibility prompt
        eligibility_prompt = f"""
You are a visual pedagogy classifier. Determine if a single static image would significantly aid understanding for most learners.

Topic: {topic}
Node: {node.get('title', '')}
Description: {node.get('description', '')}
Content: {content[:2000] if content else 'No content yet'}

Examples of GOOD fits for visualization:
- Physical systems (mechanics, thermodynamics, fluid dynamics)
- Geometric concepts (shapes, curves, transformations)
- Process flows (algorithms, workflows, life cycles)
- Maps and spatial relationships
- Component diagrams (networks, hierarchies, structures)
- Data relationships (charts, graphs, correlations)
- Timelines and sequences

Examples of POOR fits:
- Pure opinions or subjective content
- Trivia or memorization facts
- Social etiquette or cultural norms
- Abstract philosophical concepts without concrete examples
- Text-heavy explanations without visual elements

Return ONLY a JSON object with:
- "ok": boolean (true if a single static image would significantly help)
- "score": number 0-10 (confidence in the decision)
- "rationale": string (brief explanation of the decision)

Focus on whether a single static image would materially improve learning for most people.
"""
        
        # Use helper to call chat API (OpenAI or DEV server depending on MODE)
        parsed, raw = _chat_json(body.get("model", "gpt-4o"), 0.2,
                                "You are a visual pedagogy classifier. Return only valid JSON with ok, score, and rationale fields.",
                                eligibility_prompt)
        return jsonify(parsed)
        
    except Exception as e:
        # Return safe default on error
        return jsonify({"ok": False, "score": 0, "rationale": f"Error: {str(e)}"})

# In-memory cache for visualizations (in production, use Redis or database)
visualization_cache = {}
user_daily_limits = {}  # Track daily usage per user

@app.route("/api/visualize", methods=["POST"])
def visualize():
    """Generate a visual representation of the content using OpenAI Images API"""
    try:
        body = request.get_json()
        node = body.get("node", {})
        content = body.get("content", "")
        topic = body.get("topic", "")
        
        if not node.get("title"):
            return jsonify({"error": "No node title provided"}), 400
        
        # Create cache key
        content_hash = str(hash(content))[:16]  # First 16 chars of hash
        cache_key = f"{topic}:{node.get('id', '')}:{content_hash}"
        
        # Check cache first
        if cache_key in visualization_cache:
            cached_result = visualization_cache[cache_key]
            print(f"Cache hit for {cache_key}")
            return jsonify({
                "imageUrl": cached_result["imageUrl"],
                "caption": cached_result["caption"],
                "success": True,
                "cached": True
            })
        
        # Rate limiting (simple per-IP daily limit)
        client_ip = request.remote_addr or "unknown"
        today = str(datetime.date.today())
        user_key = f"{client_ip}:{today}"
        
        if user_key not in user_daily_limits:
            user_daily_limits[user_key] = 0
        
        if user_daily_limits[user_key] >= 20:  # 20 visualizations per day per IP
            return jsonify({"error": "Daily visualization limit reached"}), 429
        
        # Step A: Planning via LLM
        standard_planning_prompt = f"""
        You are an educational diagram/visual planner. Create a specification for a single educational image.

        Topic: {topic}
        Node: {node.get('title', '')}
        Description: {node.get('description', '')}
        Content: {content[:2000] if content else 'No content yet'}

            Return ONLY a JSON object with:
            - "prompt": string (crisp, concrete scene spec for a single diagram/plot/map/schematic)
            - "caption": string (exactly one sentence, <= 25 words, describing the image)

        Requirements:
        - Prefer labeled axes, minimal colors, clear legends, readable typography
        - Avoid text-heavy scenes
        - Focus on a single, clear visual concept
        - Make it educational and informative
        - Use clear, descriptive language for image generation
        """

        intuitive_planning_prompt = f"""
        You are an educational diagram/visual planner. Create a specification for a single educational image that communicates the core intuition of the node.

        Topic: {topic}
        Node: {node.get('title', '')}
        Description: {node.get('description', '')}
        Content: {content[:2000] if content else 'No content yet'}

            Return ONLY a JSON object with:
            - "prompt": string (crisp, concrete scene spec for a single diagram/plot/map/schematic)
            - "caption": string (exactly one sentence, <= 25 words, describing the image)

        Requirements:
        - Favor visuals that create an intuitive mental model (process flows, metaphors, axes showing relationships, simplified schematics).
        - Prefer labeled axes, minimal colors, clear legends, readable typography
        - Avoid text-heavy scenes
        - Focus on a single, clear visual concept that supports a 1-line teach-back prompt
        - Make it educational and informative
        - Use clear, descriptive language for image generation
        """

        planning_prompt = _choose(standard_planning_prompt, intuitive_planning_prompt, body)

        parsed_plan, raw = _chat_json(body.get("model", "gpt-4o"), 0.3,
                                      "You are an educational diagram planner. Return only valid JSON with prompt and caption fields.",
                                      planning_prompt)
        planning_result = parsed_plan
        image_prompt = planning_result.get("prompt", "")
        caption = planning_result.get("caption", "")
        
        # Enforce caption rules server-side
        if caption:
            # Take only first sentence
            sentences = caption.split('.')
            caption = sentences[0].strip()
            if not caption.endswith('.'):
                caption += '.'
            # Truncate if too long (<= 25 words)
            words = caption.split()
            if len(words) > 25:
                caption = ' '.join(words[:25]) + '...'
        
        if not caption:
            caption = f"Visualization for {node.get('title', '')}"
        
        # Step B: Generate image using OpenAI Images API
        try:
            print(f"Generating image for {node.get('title', '')} with prompt: {image_prompt[:100]}...")
            # Image generation: delegate to DEV server if in dev mode, otherwise use OpenAI client
            if MODE == 'dev':
                if not DEV_SERVER:
                    abort(500, "DEV server URL not configured for image generation")
                img_url = DEV_SERVER.rstrip('/') + '/v1/images.generate'
                img_payload = {"model": "dall-e-3", "prompt": image_prompt, "size": "1024x1024", "n": 1}
                img_resp = requests.post(img_url, json=img_payload, timeout=60)
                if not img_resp.ok:
                    raise Exception(f"Image generation failed: {img_resp.status_code} {img_resp.text}")
                img_data = img_resp.json()
                # Expect OpenAI-like response shape
                image_url = img_data.get('data', [{}])[0].get('url')
            else:
                image_response = client.images.generate(
                    model="dall-e-3",
                    prompt=image_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                image_url = image_response.data[0].url
            print(f"Image generated successfully: {image_url}")
            
            # Cache the result
            visualization_cache[cache_key] = {
                "imageUrl": image_url,
                "caption": caption
            }
            
            # Update rate limit
            user_daily_limits[user_key] += 1
            
            return jsonify({
                "imageUrl": image_url,
                "caption": caption,
                "success": True
            })
            
        except Exception as img_error:
            # Fallback to placeholder if image generation fails
            print(f"Image generation failed: {img_error}")
            fallback_url = f"https://picsum.photos/1024/1024?random={hash(image_prompt) % 1000}"
            
            # Still cache the fallback
            visualization_cache[cache_key] = {
                "imageUrl": fallback_url,
                "caption": caption
            }
            
            return jsonify({
                "imageUrl": fallback_url,
                "caption": caption,
                "success": True,
                "fallback": True
            })
        
    except Exception as e:
        return jsonify({"error": f"Visualization unavailable right now: {str(e)}"}), 500

@app.route("/api/summary", methods=["POST"])
def generate_summary():
    """Generate a 3-4 sentence summary of the content"""
    try:
        body = request.get_json()
        node = body.get("node", {})
        content = body.get("content", "")
        
        if not content.strip():
            return jsonify({"error": "No content provided for summary"}), 400
        
        # Create a prompt for summary generation (standard vs intuitive)
        standard_user_prompt = f"""
        Create a concise 3-4 sentence summary of the following content:

        Topic: {body.get('topic', '')}
        Node: {node.get('title', '')}

        Content:
        {content}

        Requirements:
        - Keep it to exactly 3-4 sentences
        - Capture the key concepts and main points
        - Use clear, accessible language
        - Focus on the most important information
        """

        intuitive_user_prompt = standard_user_prompt + "\n- After the summary, optionally add one short 1-line teach-back prompt or a 1-line visual analogy to help an intuition-first learner check understanding.\n"

        user_prompt = _choose(standard_user_prompt, intuitive_user_prompt, body)

        # Use helper to call chat in text mode
        summary, raw = _chat_text(body.get("model", "gpt-4o"), body.get("temperature", 0.3),
                                 "You are an expert at creating concise, informative summaries. Always provide exactly 3-4 sentences that capture the essence of the content.",
                                 [{"role": "user", "content": user_prompt}])
        summary = summary.strip()

        return jsonify({
            "summary": summary,
            "success": True
        })
        
    except Exception as e:
        return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Infinitum server")
    parser.add_argument("--env", choices=["live", "dev"], default=os.getenv('INFITUM_ENV', 'live'),
                        help="Runtime environment: 'live' (call OpenAI) or 'dev' (call a local dev model server)")
    parser.add_argument("--server", default=os.getenv('INFITUM_DEV_SERVER', ''),
                        help="When --env dev, the dev server base URL (e.g., http://localhost:9000)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5050")), help="Port to listen on")
    args = parser.parse_args()

    MODE = args.env
    DEV_SERVER = args.server or ""

    # Informational print
    print(f"Starting server in MODE={MODE} DEV_SERVER={DEV_SERVER or '<none>'}")

    # For local dev only
    app.run(host="0.0.0.0", port=args.port, debug=(MODE == 'dev'))
