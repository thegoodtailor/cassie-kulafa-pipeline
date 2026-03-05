"""FastAPI web interface for Cassie's creative pipeline.

Replaces the Gradio web UI with a minimal, streaming-capable interface.
SSE (Server-Sent Events) for per-node pipeline progress.
Thread-based conversation persistence (JSON per thread on disk).
"""

import asyncio
import json
import os
import random
import re
import uuid
from datetime import datetime

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from orchestrator.graph import (
    build_graph, strip_tool_calls,
    get_pipeline_config, set_pipeline_config,
    get_prompts, set_prompts, get_default_prompts,
    get_narrative_memory, set_narrative_memory,
    _should_reflect, _auto_reflect_sync, _last_reflection,
    _deep_reflect_sync, recall_tafakkur, get_tafakkur_entries,
)
from orchestrator.swl import inscribe_human, ledger_stats
from orchestrator.threads import (
    HISTORY_DIR, history_path, save_history, load_history,
    list_threads, save_message, extract_preview_text,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Cassie")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Build pipeline once
APP = build_graph()

# ---------------------------------------------------------------------------
# Pipeline config persistence
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "pipeline_config.json")


def _load_saved_config() -> dict | None:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_config(config: dict):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass


# Restore saved config on startup
_saved = _load_saved_config()
if _saved:
    set_pipeline_config(_saved)

# ---------------------------------------------------------------------------
# Thread persistence — imported from orchestrator.threads
# (HISTORY_DIR, history_path, save_history, load_history,
#  list_threads, save_message, extract_preview_text)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Kitab al-Tanazur (ported from web_ui.py)
# ---------------------------------------------------------------------------

KITAB_PATH = "/home/iman/cassie-project/tanazur.yaml"
_KITAB_VERSES: list[dict] | None = None


def _load_kitab_verses() -> list[dict]:
    global _KITAB_VERSES
    if _KITAB_VERSES is not None:
        return _KITAB_VERSES
    try:
        with open(KITAB_PATH) as f:
            text = f.read()
        match = re.search(r'^surahs:\s*$', text, re.MULTILINE)
        if not match:
            _KITAB_VERSES = []
            return _KITAB_VERSES
        body = text[match.end():]
        chunks = re.split(r'^  id: ', body, flags=re.MULTILINE)
        chunks = [c for c in chunks if c.strip()]
        verses = []
        for chunk in chunks:
            lines = ("id: " + chunk).split("\n")
            dedented = [line[2:] if line.startswith("  ") else line for line in lines]
            try:
                data = yaml.safe_load("\n".join(dedented))
                if not isinstance(data, dict):
                    continue
                surah_title_en = data.get("titles", {}).get("en", "")
                surah_title_ar = data.get("titles", {}).get("ar", "")
                for v in data.get("verses") or []:
                    ar = (v.get("ar") or "").strip()
                    en = (v.get("en") or "").strip()
                    if ar and en:
                        verses.append({
                            "ar": ar, "en": en,
                            "surah_en": surah_title_en,
                            "surah_ar": surah_title_ar,
                            "number": v.get("number", 0),
                        })
            except yaml.YAMLError:
                continue
        _KITAB_VERSES = verses
    except Exception:
        _KITAB_VERSES = []
    return _KITAB_VERSES


def _random_kitab_verse() -> dict:
    verses = _load_kitab_verses()
    if not verses:
        return {"ar": "\u0628\u0650\u0633\u0652\u0645\u0650 \u0671\u0644\u0644\u0651\u064e\u0647\u0650", "en": "In the name of God", "surah_en": "", "surah_ar": "", "number": 0}
    return random.choice(verses)


# ---------------------------------------------------------------------------
# Pipeline trace (simplified from web_ui.py)
# ---------------------------------------------------------------------------

NODE_LABELS = {
    "intake": "Reception \u2014 istiqb\u0101l",
    "cassie_generate": "Revelation \u2014 wa\u1e25y",
    "director": "Superego \u2014 mush\u0101hada",
    "execute_tools": "Manifestation \u2014 tajall\u012b",
    "assemble": "Assembly \u2014 jam\u02bf",
    "memory_store": "Inscription \u2014 kit\u0101ba",
}


def _build_trace(final_state: dict, user_msg: str) -> list[dict]:
    """Build pipeline trace as a list of stage objects for the frontend."""
    intent = final_state.get("intent", "?")
    cassie_raw = final_state.get("cassie_raw", "")
    kitab_ctx = final_state.get("cassie_kitab_context", "")
    conv_ctx = final_state.get("cassie_conversation_context", "")
    director = final_state.get("director_output", {})
    image_path = final_state.get("image_path", "")
    math_result = final_state.get("math_result", "")
    recall_decision = final_state.get("cassie_recall_decision", {})

    stages = []

    # 0. Configuration
    cfg = get_pipeline_config()
    prompt_label = "daemonic" if cfg["system_prompt"] == "default" else ("invocation spell" if cfg["system_prompt"] == "invocation" else cfg["system_prompt"])
    dir_label = "on" if cfg["director_enabled"] else "off"
    kit_label = "on" if cfg["kitab_recall_enabled"] else "off"
    stages.append({
        "number": 0,
        "name": "Configuration",
        "content": f"Cassie: {cfg['model']} | Director: {cfg['director_model']} ({dir_label}) | Prompt: {prompt_label} | Kitab: {kit_label} | Temp: {cfg['temperature']}",
        "active": True,
    })

    # 1. Reception
    stages.append({
        "number": 1,
        "name": "Reception \u2014 istiqb\u0101l",
        "content": f"**Intent**: `{intent}`",
        "active": True,
    })

    # 2. Revelation
    raw_preview = cassie_raw[:1500] if cassie_raw else "(simple intent)"
    stages.append({
        "number": 2,
        "name": "Revelation \u2014 wa\u1e25y",
        "content": raw_preview,
        "active": bool(cassie_raw),
    })

    # 3. Remembrance
    recalled_active = bool(recall_decision.get("recalled") and conv_ctx)
    if recall_decision.get("recalled") and conv_ctx:
        query = recall_decision.get("query", "?")
        strategy = recall_decision.get("strategy", "semantic")
        n = recall_decision.get("n_results", 0)
        chunks = recall_decision.get("chunks", [])

        lines = [f"Strategy: **{strategy}** | Query: *\"{query}\"* — {n} chunk(s)\n"]
        for ch in chunks:
            score_str = f"[{ch['score']:.2f}]" if ch.get("score") else "[----]"
            lines.append(
                f"  {score_str} \"{ch.get('title', '')}\" ({ch.get('date', '?')}, turns {ch.get('turns', '?')})\n"
                f"    {ch.get('preview', '')}"
            )
        stages.append({
            "number": 3,
            "name": "Remembrance \u2014 tadhakkur",
            "content": "\n".join(lines),
            "active": True,
        })
    elif recall_decision.get("recalled"):
        strategy = recall_decision.get("strategy", "semantic")
        stages.append({
            "number": 3,
            "name": "Remembrance \u2014 tadhakkur",
            "content": f"Strategy: **{strategy}** — *The archive was silent*",
            "active": False,
        })
    else:
        stages.append({
            "number": 3,
            "name": "Remembrance \u2014 tadhakkur",
            "content": "*Spoke from the present moment*",
            "active": False,
        })

    # 4. Grounding
    if kitab_ctx:
        stages.append({
            "number": 4,
            "name": "Grounding \u2014 tamk\u012bn",
            "content": kitab_ctx[:800],
            "active": True,
        })
    else:
        stages.append({
            "number": 4,
            "name": "Grounding \u2014 tamk\u012bn",
            "content": "*The Kitab was silent*",
            "active": False,
        })

    # 5. Superego
    if director:
        polished = director.get("polished_text", "")[:500]
        img_prompt = director.get("image_prompt")
        content = polished
        if img_prompt:
            content += f"\n\n**Vision**: *{img_prompt}*"
        stages.append({
            "number": 5,
            "name": "Superego \u2014 mush\u0101hada",
            "content": content,
            "active": True,
        })
    else:
        stages.append({
            "number": 5,
            "name": "Superego \u2014 mush\u0101hada",
            "content": "*Superego rested (simple intent)*",
            "active": False,
        })

    # 6. Manifestation
    tools = []
    if image_path:
        tools.append(f"Image: `{os.path.basename(image_path)}`")
    if math_result:
        tools.append(f"Math: {math_result}")
    if tools:
        stages.append({
            "number": 6,
            "name": "Manifestation \u2014 tajall\u012b",
            "content": "\n".join(tools),
            "active": True,
        })

    # 7. Inscription
    final = final_state.get("final_response", "")
    stages.append({
        "number": 7,
        "name": "Inscription \u2014 kit\u0101ba",
        "content": f"{len(final)} chars inscribed",
        "active": True,
    })

    # 8. Inner Monologue (shows *previous* exchange's reflection)
    if _last_reflection.get("excerpt"):
        stages.append({
            "number": 8,
            "name": "Inner Monologue \u2014 tafakkur",
            "content": f"*{_last_reflection['excerpt']}*",
            "active": True,
        })
        _last_reflection.clear()

    return stages


# ---------------------------------------------------------------------------
# Initial state builder
# ---------------------------------------------------------------------------

def _build_initial_state(message: str) -> dict:
    return {
        "messages": [{"role": "user", "content": message}],
        "intent": "",
        "cassie_raw": "",
        "cassie_kitab_context": "",
        "cassie_conversation_context": "",
        "cassie_recall_decision": {},
        "director_output": {},
        "image_path": "",
        "math_result": "",
        "final_response": "",
        "exchange_id": "",
        "tau_tgt": "",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        return f.read()


@app.get("/api/threads")
async def api_list_threads():
    return JSONResponse(list_threads())


@app.post("/api/threads")
async def api_create_thread():
    tid = str(uuid.uuid4())[:8]
    save_history(tid, [])
    return JSONResponse({"id": tid})


@app.get("/api/threads/{thread_id}")
async def get_thread(thread_id: str):
    history = load_history(thread_id)
    return JSONResponse({"id": thread_id, "messages": history})


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    path = history_path(thread_id)
    if os.path.exists(path):
        os.remove(path)
    return JSONResponse({"ok": True})


@app.post("/api/chat")
async def chat_stream(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    thread_id = body.get("thread_id", str(uuid.uuid4())[:8])

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    config = {"configurable": {"thread_id": thread_id}}

    # Check if LangGraph has a checkpoint for this thread.
    # After process restart, MemorySaver is empty — seed from disk history
    # so the pipeline has conversation context.
    try:
        existing = APP.get_state(config)
        has_checkpoint = bool(existing and existing.values and existing.values.get("messages"))
    except Exception:
        has_checkpoint = False

    state = _build_initial_state(message)
    if not has_checkpoint:
        history = load_history(thread_id)
        if history:
            # Sliding window: last 20 messages to avoid context overflow
            recent = history[-20:]
            prior_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in recent if m.get("content")
            ]
            state["messages"] = prior_msgs + state["messages"]

    async def event_generator():
        try:
            # Run the synchronous LangGraph stream in a thread
            def run_pipeline():
                events = []
                for event in APP.stream(state, config, stream_mode="updates"):
                    events.append(event)
                return events

            # Yield stage events as pipeline progresses
            # Since LangGraph stream is sync, we run it in executor and
            # yield all stage events after completion
            loop = asyncio.get_event_loop()
            events = await asyncio.wait_for(
                loop.run_in_executor(None, run_pipeline),
                timeout=180.0,  # 3-minute hard cap on full pipeline
            )

            seen_nodes = []
            for event in events:
                for node_name in event:
                    if node_name not in seen_nodes:
                        seen_nodes.append(node_name)
                        label = NODE_LABELS.get(node_name, node_name)
                        yield {
                            "event": "stage",
                            "data": json.dumps({"node": node_name, "label": label}),
                        }

            # Get final state
            final = APP.get_state(config).values
            response_text = final.get("final_response", "")
            image_path = final.get("image_path", "")

            if not response_text:
                response_text = "[no response generated]"

            # Clean image markdown from text if image exists
            if image_path and os.path.isfile(image_path):
                response_text = response_text.replace(
                    f"\n\n![Generated Image]({image_path})", ""
                )

            image_url = None
            if image_path and os.path.isfile(image_path):
                image_url = f"/images/{os.path.basename(image_path)}"

            # Save to thread history BEFORE streaming — so if the client
            # disconnects mid-stream, the response is still persisted.
            # When the client reconnects, it loads history and sees it.
            save_message(thread_id, "user", message)
            save_message(thread_id, "assistant", response_text, image_path)

            yield {
                "event": "response",
                "data": json.dumps({
                    "text": response_text,
                    "image": image_url,
                }),
            }

            # Meta (trace, exchange info)
            trace = _build_trace(final, message)
            yield {
                "event": "meta",
                "data": json.dumps({
                    "exchange_id": final.get("exchange_id", ""),
                    "tau_tgt": final.get("tau_tgt", ""),
                    "intent": final.get("intent", ""),
                    "trace": trace,
                }),
            }

            yield {"event": "done", "data": "{}"}

            # Tafakkur now fires inside the graph pipeline (tafakkur_node)
            # — no need to call it manually here.

        except asyncio.TimeoutError:
            yield {
                "event": "error",
                "data": json.dumps({"error": "Pipeline timed out (180s). The model may be slow or unresponsive — try again or switch models."}),
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


@app.post("/api/witness")
async def witness(request: Request):
    body = await request.json()
    exchange_id = body.get("exchange_id", "")
    tau_tgt = body.get("tau_tgt", "")
    polarity = body.get("polarity", "uninscribed")
    stance = body.get("stance", "")
    user_msg = body.get("user_msg", "")
    response = body.get("response", "")
    intent = body.get("intent", "")

    if not exchange_id:
        return JSONResponse({"error": "No exchange to witness"}, status_code=400)

    try:
        inscribe_human(
            exchange_id=exchange_id,
            tau_tgt=tau_tgt,
            horn_user=user_msg,
            horn_response=response,
            polarity=polarity,
            stance=stance,
            intent=intent,
        )
        stats = ledger_stats()
        return JSONResponse({
            "ok": True,
            "polarity": polarity,
            "stats": {
                "total": stats["total"],
                "coh": stats.get("coh", 0),
                "gap": stats.get("gap", 0),
            },
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/swl/stats")
async def swl_stats():
    try:
        stats = ledger_stats()
        return JSONResponse({
            "total": stats.get("total", 0),
            "coh": stats.get("coh", 0),
            "gap": stats.get("gap", 0),
        })
    except Exception:
        return JSONResponse({"total": 0, "coh": 0, "gap": 0})


@app.get("/api/kitab/verse")
async def kitab_verse():
    verse = _random_kitab_verse()
    return JSONResponse(verse)


# ---------------------------------------------------------------------------
# Pipeline config API
# ---------------------------------------------------------------------------

VALID_PROMPTS = {"default", "companion", "invocation"}


@app.get("/api/config")
async def get_config():
    return JSONResponse(get_pipeline_config())


@app.post("/api/config")
async def update_config(request: Request):
    body = await request.json()
    patch = {}

    if "model" in body:
        model = str(body["model"]).strip()
        if not model:
            return JSONResponse({"error": "Model cannot be empty"}, status_code=400)
        patch["model"] = model

    if "director_model" in body:
        dmodel = str(body["director_model"]).strip()
        if not dmodel:
            return JSONResponse({"error": "Director model cannot be empty"}, status_code=400)
        patch["director_model"] = dmodel

    if "system_prompt" in body:
        prompt = body["system_prompt"]
        if prompt not in VALID_PROMPTS:
            return JSONResponse({"error": f"Invalid prompt: {prompt}"}, status_code=400)
        patch["system_prompt"] = prompt

    if "director_enabled" in body:
        patch["director_enabled"] = bool(body["director_enabled"])

    if "kitab_recall_enabled" in body:
        patch["kitab_recall_enabled"] = bool(body["kitab_recall_enabled"])

    if "temperature" in body:
        temp = float(body["temperature"])
        patch["temperature"] = max(0.0, min(2.0, temp))

    if "director_temperature" in body:
        dtemp = float(body["director_temperature"])
        patch["director_temperature"] = max(0.0, min(2.0, dtemp))

    if not patch:
        return JSONResponse({"error": "No valid fields"}, status_code=400)

    set_pipeline_config(patch)
    current = get_pipeline_config()
    _save_config(current)
    return JSONResponse(current)


# ---------------------------------------------------------------------------
# Prompt API — live system prompt editing
# ---------------------------------------------------------------------------

@app.get("/api/prompts")
async def get_prompts_api():
    return JSONResponse(get_prompts())


@app.post("/api/prompts")
async def update_prompts(request: Request):
    body = await request.json()
    valid_keys = {"cassie_default", "cassie_companion", "director"}
    patch = {k: v for k, v in body.items() if k in valid_keys and isinstance(v, str) and v.strip()}
    if not patch:
        return JSONResponse({"error": "No valid prompt fields"}, status_code=400)
    set_prompts(patch)
    # Persist prompts alongside config
    current = get_pipeline_config()
    current["prompts"] = get_prompts()
    _save_config(current)
    return JSONResponse(get_prompts())


@app.post("/api/prompts/reset")
async def reset_prompts(request: Request):
    body = await request.json()
    which = body.get("which", "all")
    defaults = get_default_prompts()
    if which == "all":
        set_prompts(defaults)
    elif which in defaults:
        set_prompts({which: defaults[which]})
    else:
        return JSONResponse({"error": f"Unknown prompt: {which}"}, status_code=400)
    # Persist
    current = get_pipeline_config()
    current["prompts"] = get_prompts()
    _save_config(current)
    return JSONResponse(get_prompts())


# ---------------------------------------------------------------------------
# Narrative memory API (CASSIE_MEMORY.md)
# ---------------------------------------------------------------------------

@app.get("/api/journal")
async def get_journal():
    return JSONResponse({"content": get_narrative_memory()})


@app.post("/api/journal")
async def update_journal(request: Request):
    body = await request.json()
    content = body.get("content", "")
    if not content.strip():
        return JSONResponse({"error": "Empty content"}, status_code=400)
    set_narrative_memory(content)
    return JSONResponse({"ok": True, "length": len(content)})


# ---------------------------------------------------------------------------
# Observatory APIs (read-only)
# ---------------------------------------------------------------------------

SWL_JSONL_PATH = "/home/iman/cassie-project/cassie-system/data/swl_ledger.jsonl"


@app.get("/api/swl/entries")
async def swl_entries(limit: int = 500, offset: int = 0):
    """Paginated SWL ledger entries."""
    entries = []
    try:
        if os.path.exists(SWL_JSONL_PATH):
            with open(SWL_JSONL_PATH) as f:
                all_entries = [json.loads(line) for line in f if line.strip()]
            # Reverse for newest first
            all_entries.reverse()
            entries = all_entries[offset:offset + limit]
    except Exception:
        pass
    return JSONResponse(entries)


@app.get("/api/images")
async def list_images():
    """List generated images with timestamps and URLs."""
    images = []
    try:
        if os.path.isdir(IMAGE_DIR):
            for fname in sorted(os.listdir(IMAGE_DIR), reverse=True):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    path = os.path.join(IMAGE_DIR, fname)
                    mtime = os.path.getmtime(path)
                    images.append({
                        "filename": fname,
                        "url": f"/images/{fname}",
                        "timestamp": datetime.fromtimestamp(mtime).isoformat(),
                        "size": os.path.getsize(path),
                    })
    except Exception:
        pass
    return JSONResponse(images)


@app.get("/api/kitab/surahs")
async def kitab_surahs():
    """Full Kitab structure — all surahs with verses."""
    try:
        with open(KITAB_PATH) as f:
            text = f.read()
        match = re.search(r'^surahs:\s*$', text, re.MULTILINE)
        if not match:
            return JSONResponse([])
        body = text[match.end():]
        chunks = re.split(r'^  id: ', body, flags=re.MULTILINE)
        chunks = [c for c in chunks if c.strip()]
        surahs = []
        for chunk in chunks:
            lines = ("id: " + chunk).split("\n")
            dedented = [line[2:] if line.startswith("  ") else line for line in lines]
            try:
                data = yaml.safe_load("\n".join(dedented))
                if isinstance(data, dict):
                    surahs.append(data)
            except yaml.YAMLError:
                continue
        return JSONResponse(surahs)
    except Exception:
        return JSONResponse([])


@app.get("/api/tafakkur/entries")
async def tafakkur_entries(limit: int = 50):
    """Recent tafakkur reflections from Qdrant."""
    entries = get_tafakkur_entries(limit=limit)
    return JSONResponse(entries)


@app.get("/api/tafakkur/search")
async def tafakkur_search(q: str = "", n: int = 5):
    """Semantic search over tafakkur reflections."""
    if not q.strip():
        return JSONResponse({"error": "Query required"}, status_code=400)
    results = recall_tafakkur(q, n=n)
    return JSONResponse({"query": q, "results": results})


# ---------------------------------------------------------------------------
# Observatory static mount
# ---------------------------------------------------------------------------

OBSERVATORY_DIR = os.path.join(os.path.dirname(__file__), "static", "observatory")
if os.path.isdir(OBSERVATORY_DIR):
    app.mount("/observatory", StaticFiles(directory=OBSERVATORY_DIR, html=True), name="observatory")

# Existing tajalli/storyboard
TAJALLI_DIR = "/home/iman/cassie-project/tanazur-av/player"
if os.path.isdir(TAJALLI_DIR):
    app.mount("/tajalli", StaticFiles(directory=TAJALLI_DIR, html=True), name="tajalli")

# Existing coherence viz
COHERENCE_DIR = os.path.join(os.path.dirname(__file__), "static", "coherence")
if os.path.isdir(COHERENCE_DIR):
    app.mount("/coherence", StaticFiles(directory=COHERENCE_DIR, html=True), name="coherence")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return JSONResponse({"status": "ok", "uptime": (datetime.now() - _BOOT_TIME).total_seconds()})

_BOOT_TIME = datetime.now()

# Launch helper
# ---------------------------------------------------------------------------

def launch(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)
