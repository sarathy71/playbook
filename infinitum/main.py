import os
import json
import uuid
import datetime
from flask import Flask, request, jsonify, render_template, abort
from dotenv import load_dotenv
import requests
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")  # optional

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

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
        "Be concise and clear. Use Markdown and LaTeX for math when appropriate. "
        "For math expressions, use $...$ for inline math or $$...$$ for display math. "
        "Do not surround TeX with normal parentheses or brackets, and do not put math inside code fences." + level_instruction
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
            "Given a short text selection, produce 1-3 child sections that explain and deepen understanding "
            "of that selection. Do NOT reference the parent topic, breadcrumb, or surrounding context."
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
        
        response = client.chat.completions.create(
            model=body.get("model", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are a visual pedagogy classifier. Return only valid JSON with ok, score, and rationale fields."},
                {"role": "user", "content": eligibility_prompt}
            ],
            temperature=0.2,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        eligibility_data = json.loads(result)
        
        return jsonify(eligibility_data)
        
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
        planning_prompt = f"""
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
        
        planning_response = client.chat.completions.create(
            model=body.get("model", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are an educational diagram planner. Return only valid JSON with prompt and caption fields."},
                {"role": "user", "content": planning_prompt}
            ],
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        planning_result = json.loads(planning_response.choices[0].message.content)
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
        
        # Create a prompt for summary generation
        user_prompt = f"""
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
        
        response = client.chat.completions.create(
            model=body.get("model", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are an expert at creating concise, informative summaries. Always provide exactly 3-4 sentences that capture the essence of the content."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=body.get("temperature", 0.3),
            max_tokens=300
        )
        
        summary = response.choices[0].message.content.strip()
        
        return jsonify({
            "summary": summary,
            "success": True
        })
        
    except Exception as e:
        return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500

if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5050")), debug=True)
