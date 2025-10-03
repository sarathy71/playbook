import os
import json
import uuid
from flask import Flask, request, jsonify, render_template, abort
from dotenv import load_dotenv
import requests

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")  # optional

app = Flask(__name__)

SYSTEM_PROMPT = (
    "You are an expert curriculum designer and information architect. "
    "Given a Topic and constraints, produce a clean, hierarchical table of contents. "
    "Focus on clarity, progressive depth, and coverage of the domain. Avoid fluff, "
    "keep titles concise, and add 1-sentence descriptions only when valuable."
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

def _chat_json(model, temperature, system, user):
    """Call OpenAI chat.completions expecting JSON content."""
    if not OPENAI_API_KEY:
        abort(400, "OPENAI_API_KEY is not configured on the server.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
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
    if not OPENAI_API_KEY:
        abort(400, "OPENAI_API_KEY is not configured on the server.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
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

    user_prompt = f"""
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

    parsed, raw = _chat_json(model, temperature, SYSTEM_PROMPT, user_prompt)
    toc = _ensure_ids(parsed.get("toc", []))
    return jsonify({"toc": toc, "model": raw.get("model"), "tokens": raw.get("usage")})

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

    user_prompt = f"""
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

    parsed, raw = _chat_json(model, temperature, SYSTEM_PROMPT, user_prompt)
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
    level = int(body.get("level", 5))  # Default to level 5 if not specified
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
    
    level_instruction = level_descriptions.get(level, level_descriptions[5])

    user_prompt = f"""
Write a clear, well-structured explanation (250–450 words) for the selected outline item. Use Markdown and LaTeX for math.

Global Topic: {topic}
Audience: {audience}
Proficiency Level: {level}/10 - {level_instruction}
Depth preference (context): {depth}
Approx sections per level (context): {sections}

Breadcrumb (root->current): {' > '.join([p.get('title','') for p in path])}
Current Item: {node.get('title','')}
Current Item ID: {node.get('id','')}

Style:
- Use paragraphs and short bullet lists where helpful.
- Inline math: $...$ ; display math: $$...$$
- Match the requested proficiency level exactly: {level_instruction}
- No JSON.
""".strip()

    content, raw = _chat_text(model, temperature,
                              "You explain concepts clearly with examples. Use Markdown and LaTeX for math.",
                              [{"role": "user", "content": user_prompt}])
    return jsonify({"content": content, "model": raw.get("model"), "tokens": raw.get("usage")})

@app.post("/api/chat")
def api_chat():
    body = request.get_json(force=True, silent=True) or {}
    topic = body.get("topic", "").strip()
    node = body.get("node", {})
    path = body.get("path", [])
    audience = body.get("audience", "general")
    model = body.get("model", "gpt-4.1-mini")
    temperature = float(body.get("temperature", 0.3))
    question = (body.get("question") or "").strip()
    context = body.get("context") or ""  # the 'Read' content under the node
    history = body.get("history") or []  # [{role, content}...]
    level = body.get("level")  # Optional level for level-aware responses

    if not topic or not node or not question:
        abort(400, "Missing topic, node, or question.")

    # Level-aware system prompt
    level_instruction = ""
    if level:
        level_descriptions = {
            1: "Answer as if speaking to a curious child: use simple analogies, avoid jargon, focus on understanding.",
            2: "Answer for someone with basic knowledge: use simple concepts, minimal technical terms.",
            3: "Answer for an introductory learner: use key concepts and intuition, basic terminology.",
            4: "Answer for an intermediate learner: balanced explanation with some technical details.",
            5: "Answer for an informed learner: comprehensive overview with good balance of intuition and technical details.",
            6: "Answer for an advanced learner: detailed explanation with mathematical concepts, technical terminology.",
            7: "Answer for an expert: rigorous treatment with derivations, formal definitions, advanced mathematics.",
            8: "Answer for a specialist: formal treatment with proofs, advanced formalism, specialized terminology.",
            9: "Answer for a researcher: cutting-edge concepts, advanced formalism, research-level depth.",
            10: "Answer at maximum expertise level: complete formalism, rigorous proofs, advanced mathematics."
        }
        level_instruction = f" Match the requested proficiency level {level}/10: {level_descriptions.get(level, level_descriptions[5])}."

    system = (
        "Answer strictly using the provided section context unless the question is generic. "
        "Be concise and clear. Use Markdown and LaTeX for math when appropriate." + level_instruction
    )

    messages = [
        {
            "role": "user",
            "content": f"""Context (from the selected item):
{context}

Config:
- Topic: {topic}
- Audience: {audience}
- Breadcrumb: {' > '.join([p.get('title','') for p in path])}
- Item: {node.get('title','')} (ID: {node.get('id','')})

You will now answer questions about this item only.
"""
        }
    ]
    # append last few turns
    for m in history[-6:]:
        if m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m.get("content","")})
    # the new question
    messages.append({"role": "user", "content": question})

    answer, raw = _chat_text(model, temperature, system, messages)
    return jsonify({"answer": answer, "model": raw.get("model"), "tokens": raw.get("usage")})

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
    
    if not topic or not node or not selection:
        abort(400, "Missing topic, node, or selection.")
    
    selection_text = selection.get("text", "").strip()
    if len(selection_text) < 8:
        abort(400, "Selection too short. Please select at least 8 characters.")
    if len(selection_text) > 3000:
        abort(400, "Selection too long. Please select less than 3000 characters.")

    system_prompt = (
        "You are an expert curriculum designer creating focused, educational content. "
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
""".strip()

    parsed, raw = _chat_json(model, temperature, system_prompt, user_prompt)
    
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

@app.post("/api/notebook/save")
def api_notebook_save():
    """Save a notebook to server storage (optional cloud persistence)."""
    body = request.get_json(force=True, silent=True) or {}
    notebook_id = body.get("id") or str(uuid.uuid4())
    notebook_data = body.get("notebook")
    
    if not notebook_data:
        abort(400, "Notebook data is required.")
    
    # In a real implementation, you'd save to a database
    # For now, we'll just return success with the ID
    # You could implement file-based storage or database persistence here
    
    return jsonify({
        "id": notebook_id,
        "saved": True,
        "message": "Notebook saved successfully"
    })

@app.get("/api/notebook/load")
def api_notebook_load():
    """Load a notebook from server storage."""
    notebook_id = request.args.get("id")
    if not notebook_id:
        abort(400, "Notebook ID is required.")
    
    # In a real implementation, you'd load from a database
    # For now, return a 404 since we don't have persistence implemented
    abort(404, "Notebook not found. Server-side persistence not yet implemented.")

if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5050")), debug=True)
