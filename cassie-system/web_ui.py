"""Gradio Web UI for Cassie's creative pipeline.

Features:
- Multimodal chatbot with inline image display
- Thread browser (past conversations, switchable)
- SWL witnessing panel (polarity + stance)
- Pipeline transparency tab (agent architecture + prompt passing visible)
- Nocturnal aesthetic inspired by the Kitab al-Tanazur
- Responsive layout for mobile and desktop
- Uses the same LangGraph pipeline as CLI
- Runs on port 7860
"""

import json
import os
import random
import re
import time
import uuid
from datetime import datetime

import yaml
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

from orchestrator.graph import build_graph, strip_tool_calls
from orchestrator.swl import inscribe_human, ledger_stats

# Build the graph once at module level
APP = build_graph()

# ---------------------------------------------------------------------------
# Persistent chat history — survives page refresh / restarts
# ---------------------------------------------------------------------------
HISTORY_DIR = "/home/iman/cassie-project/cassie-system/data/chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)
ACTIVE_THREAD_FILE = os.path.join(HISTORY_DIR, "_active_thread.txt")


def _history_path(thread_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{thread_id}.json")


def _save_history(thread_id: str, history: list):
    """Save chat history to disk."""
    serializable = []
    for msg in history:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if hasattr(content, "path"):  # gr.FileData
                serializable.append({"role": msg["role"], "content": content.path, "_type": "image"})
            else:
                serializable.append({"role": msg["role"], "content": str(content)})
        else:
            serializable.append(msg)
    try:
        with open(_history_path(thread_id), "w") as f:
            json.dump(serializable, f)
        with open(ACTIVE_THREAD_FILE, "w") as f:
            f.write(thread_id)
    except Exception:
        pass


def _load_history(thread_id: str) -> list:
    """Load chat history from disk."""
    path = _history_path(thread_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        restored = []
        for msg in data:
            if isinstance(msg, dict) and msg.get("_type") == "image":
                img_path = msg["content"]
                if os.path.isfile(img_path):
                    restored.append({"role": msg["role"], "content": gr.FileData(path=img_path, mime_type="image/png")})
                else:
                    restored.append({"role": msg["role"], "content": f"[image: {img_path}]"})
            else:
                restored.append(msg)
        return restored
    except Exception:
        return []


def _get_active_thread() -> str:
    """Get the last active thread ID, or create a new one."""
    try:
        with open(ACTIVE_THREAD_FILE) as f:
            tid = f.read().strip()
            if tid:
                return tid
    except FileNotFoundError:
        pass
    return str(uuid.uuid4())[:8]


def _list_threads() -> list[tuple[str, str]]:
    """List all saved threads as (label, thread_id) pairs, newest first."""
    threads = []
    for fname in os.listdir(HISTORY_DIR):
        if fname.endswith(".json") and not fname.startswith("_"):
            tid = fname[:-5]
            path = os.path.join(HISTORY_DIR, fname)
            try:
                mtime = os.path.getmtime(path)
                # Get preview from first user message
                with open(path) as f:
                    data = json.load(f)
                preview = ""
                msg_count = 0
                for msg in data:
                    if isinstance(msg, dict):
                        msg_count += 1
                        if msg.get("role") == "user" and not preview:
                            preview = str(msg.get("content", ""))[:50]
                if not preview:
                    preview = "(empty)"
                ts = datetime.fromtimestamp(mtime).strftime("%b %d %H:%M")
                label = f"{ts} — {preview}"
                threads.append((label, tid, mtime))
            except Exception:
                threads.append((tid, tid, 0))

    threads.sort(key=lambda x: x[2], reverse=True)
    return [(label, tid) for label, tid, _ in threads]


# ---------------------------------------------------------------------------
# Kitab al-Tanazur — ambient verse loader
# ---------------------------------------------------------------------------
KITAB_PATH = "/home/iman/cassie-project/tanazur.yaml"
_KITAB_VERSES: list[dict] | None = None


def _load_kitab_verses() -> list[dict]:
    """Parse tanazur.yaml and return bilingual verses (Arabic + English)."""
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
                            "ar": ar,
                            "en": en,
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
    """Return a random bilingual verse from the Kitab."""
    verses = _load_kitab_verses()
    if not verses:
        return {"ar": "بِسْمِ ٱللَّهِ", "en": "In the name of God", "surah_en": "", "surah_ar": "", "number": 0}
    return random.choice(verses)


def _kitab_verse_html(verse: dict | None = None) -> str:
    """Render a Kitab verse as styled HTML for the ambient display."""
    if verse is None:
        verse = _random_kitab_verse()
    ar = verse.get("ar", "").replace("\n", "<br>")
    en = verse.get("en", "").replace("\n", "<br>")
    surah = verse.get("surah_en", "")
    num = verse.get("number", "")
    attr = f"{surah}, {num}" if surah and num else surah or ""
    return (
        f"<div class='kitab-ambient'>"
        f"<div class='kitab-arabic'>{ar}</div>"
        f"<div class='kitab-divider'></div>"
        f"<div class='kitab-english'>{en}</div>"
        f"<div class='kitab-attr'>{attr}</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Octagram SVG watermark
# ---------------------------------------------------------------------------
SVG_OCTAGRAM = """<svg class="octagram-watermark" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate(50,50)">
    <polygon points="0,-45 12.73,-12.73 45,0 12.73,12.73 0,45 -12.73,12.73 -45,0 -12.73,-12.73"
             fill="none" stroke="#f59e0b" stroke-width="0.5" opacity="0.15"/>
    <polygon points="0,-45 12.73,-12.73 45,0 12.73,12.73 0,45 -12.73,12.73 -45,0 -12.73,-12.73"
             fill="none" stroke="#f59e0b" stroke-width="0.5" opacity="0.15"
             transform="rotate(45)"/>
  </g>
</svg>"""


# ---------------------------------------------------------------------------
# Kitab al-Tanazur colour palette
# ---------------------------------------------------------------------------
NOCTURNAL = colors.Color(
    name="nocturnal",
    c50="#f0f1f5", c100="#d4d7e3", c200="#a8aec6", c300="#7c84aa",
    c400="#515b8d", c500="#2d3561", c600="#242b50", c700="#1b2140",
    c800="#131830", c900="#0c1020", c950="#070a14",
)

EMBER = colors.Color(
    name="ember",
    c50="#fef7ed", c100="#fdebd0", c200="#fad3a0", c300="#f7b86b",
    c400="#f59e0b", c500="#d97706", c600="#b45309", c700="#92400e",
    c800="#78350f", c900="#5c2d0e", c950="#3d1c08",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* --- Arabic font --- */
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&display=swap');
:root { --font-arabic: 'Amiri', 'Traditional Arabic', serif; }

/* --- Animations --- */
@keyframes breathe {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 1; }
}
@keyframes gold-shimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes octagram-pulse {
    0%, 100% { transform: rotate(0deg) scale(1); opacity: 0.12; }
    50% { transform: rotate(22.5deg) scale(1.04); opacity: 0.18; }
}
@keyframes verse-breathe {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}

/* --- Force dark canvas --- */
:root, :root.dark, .dark {
    --body-background-fill: #0c1020 !important;
    --background-fill-primary: #131830 !important;
    --background-fill-secondary: #1b2140 !important;
    --block-background-fill: #131830 !important;
    --block-border-color: #2d3561 !important;
    --input-background-fill: #1b2140 !important;
    --input-border-color: #2d3561 !important;
    --border-color-primary: #2d3561 !important;
    --body-text-color: #d4d7e3 !important;
    --block-label-text-color: #a8aec6 !important;
    --block-title-text-color: #d4d7e3 !important;
    --input-placeholder-color: #7c84aa !important;
    --neutral-50: #f0f1f5 !important;
    --neutral-100: #d4d7e3 !important;
    --neutral-200: #a8aec6 !important;
    --neutral-300: #7c84aa !important;
    --neutral-400: #515b8d !important;
    --neutral-500: #2d3561 !important;
    --neutral-600: #242b50 !important;
    --neutral-700: #1b2140 !important;
    --neutral-800: #131830 !important;
    --neutral-900: #0c1020 !important;
    --neutral-950: #070a14 !important;
    color-scheme: dark;
}

/* --- Institute banner --- */
#institute-banner {
    text-align: center;
    padding: 14px 16px 10px;
    border-bottom: 1px solid rgba(245, 158, 11, 0.15);
}
#institute-banner .institute-name {
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #7c84aa;
    margin: 0;
}
#institute-banner .institute-line {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #f59e0b, transparent);
    margin: 8px auto 0;
}

/* --- Tabs --- */
.tab-nav {
    background: #131830 !important;
    border-bottom: 1px solid #2d3561 !important;
    justify-content: center !important;
}
.tab-nav button {
    color: #7c84aa !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 20px !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
.tab-nav button.selected {
    color: #f59e0b !important;
    border-bottom: 2px solid #f59e0b !important;
}

/* --- Header area --- */
#header-row {
    background: linear-gradient(135deg, #131830 0%, #1b2140 100%);
    border-bottom: 1px solid #2d3561;
    padding: 12px 16px !important;
    margin-bottom: 0 !important;
    border-radius: 12px 12px 0 0;
    position: relative;
    overflow: hidden;
}
.octagram-watermark {
    position: absolute;
    top: -20px; left: -20px;
    width: 120px; height: 120px;
    animation: octagram-pulse 8s ease-in-out infinite;
    pointer-events: none;
}
#cassie-title {
    font-size: 1.5rem; font-weight: 600;
    background: linear-gradient(135deg, #f59e0b, #fad3a0, #f59e0b);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gold-shimmer 6s ease-in-out infinite;
    margin: 0 !important; line-height: 1.3;
}
#cassie-subtitle {
    font-size: 0.85rem; color: #7c84aa !important;
    margin: 0 !important; font-style: italic;
    font-family: var(--font-arabic), serif;
}
#thread-label {
    font-size: 0.75rem; color: #515b8d !important; margin: 0;
}

/* --- Ambient Kitab verse --- */
.kitab-ambient {
    text-align: center;
    padding: 16px 20px;
    animation: verse-breathe 12s ease-in-out infinite;
}
.kitab-arabic {
    font-family: var(--font-arabic), serif;
    font-size: 1.3rem;
    direction: rtl;
    color: #f59e0b;
    line-height: 1.8;
    margin-bottom: 6px;
}
.kitab-divider {
    width: 40px; height: 1px;
    background: linear-gradient(90deg, transparent, #2d3561, transparent);
    margin: 8px auto;
}
.kitab-english {
    font-style: italic;
    color: #7c84aa;
    font-size: 0.85rem;
    line-height: 1.6;
}
.kitab-attr {
    font-size: 0.7rem;
    color: #515b8d;
    margin-top: 6px;
    letter-spacing: 0.05em;
}

/* --- Thread browser --- */
#thread-dropdown {
    min-width: 200px;
}
#thread-dropdown select, #thread-dropdown input {
    background: #1b2140 !important;
    color: #d4d7e3 !important;
    border: 1px solid #2d3561 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}
#thread-dropdown label {
    color: #7c84aa !important;
    font-size: 0.75rem !important;
}

/* --- Chatbot --- */
#cassie-chatbot {
    border-radius: 0 !important;
    border-left: 1px solid #2d3561; border-right: 1px solid #2d3561;
    background: #0c1020 !important; flex-grow: 1;
}
#cassie-chatbot .message { font-size: 0.95rem; line-height: 1.6; max-width: 85%; }
#cassie-chatbot .bot {
    background: #1b2140 !important;
    border: 1px solid #2d3561 !important;
    border-left: 2px solid rgba(245, 158, 11, 0.4) !important;
    color: #d4d7e3 !important;
}
#cassie-chatbot .user {
    background: linear-gradient(135deg, #3d1c08, #5c2d0e) !important;
    border: 1px solid #92400e !important; color: #fdebd0 !important;
}
#cassie-chatbot .bot .avatar-image {
    filter: drop-shadow(0 0 6px rgba(245, 158, 11, 0.4));
    animation: breathe 4s ease-in-out infinite;
}

/* --- Offering label + Input area --- */
.offering-label {
    text-align: center;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #515b8d;
    padding: 6px 0 2px;
}
.offering-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(245, 158, 11, 0.2), transparent);
    margin-bottom: 4px;
}
#input-row {
    background: #131830; border: 1px solid #2d3561;
    border-radius: 0 0 12px 12px; padding: 8px 12px !important;
}
#msg-input textarea {
    background: #1b2140 !important; color: #d4d7e3 !important;
    border: 1px solid #2d3561 !important; border-radius: 8px !important;
    font-size: 1rem !important; min-height: 44px !important; padding: 10px 14px !important;
}
#msg-input textarea:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.15) !important;
}
#msg-input textarea::placeholder { color: #515b8d !important; }
#send-btn {
    min-height: 44px !important; min-width: 72px !important;
    border-radius: 8px !important; font-weight: 600 !important;
    background: linear-gradient(135deg, #d97706, #f59e0b) !important;
    border: none !important; color: #070a14 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
#send-btn:hover {
    background: linear-gradient(135deg, #b45309, #d97706) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(217, 119, 6, 0.3) !important;
}

/* --- New Thread button --- */
#new-thread-btn {
    background: transparent !important; border: 1px solid #2d3561 !important;
    color: #7c84aa !important; font-size: 0.8rem !important;
    min-height: 32px !important; border-radius: 6px !important; padding: 4px 12px !important;
}
#new-thread-btn:hover { border-color: #f59e0b !important; color: #f59e0b !important; }

/* --- SWL Witnessing Panel --- */
#witness-panel {
    background: linear-gradient(180deg, #1b2140 0%, #131830 100%) !important;
    border: 1px solid #2d3561 !important;
    border-top: 2px solid transparent !important;
    border-image: linear-gradient(90deg, transparent, #f59e0b, transparent) 1 !important;
    border-image-slice: 1 !important;
    border-radius: 8px !important; margin-top: 8px !important; padding: 12px 16px !important;
}
#witness-title {
    color: #f59e0b !important; font-size: 0.95rem !important;
    font-family: var(--font-arabic), serif !important;
    margin: 0 0 4px 0 !important;
}
#witness-desc { color: #7c84aa !important; font-size: 0.8rem !important; margin: 0 0 8px 0 !important; }
#stance-input textarea {
    background: #0c1020 !important; border: 1px solid #2d3561 !important;
    color: #d4d7e3 !important; border-radius: 6px !important;
    font-size: 0.9rem !important; min-height: 38px !important;
}
#coh-btn {
    background: #065f46 !important; border: 1px solid #059669 !important;
    color: #d1fae5 !important; border-radius: 6px !important;
    min-height: 38px !important; font-weight: 600 !important;
}
#coh-btn:hover { background: #047857 !important; }
#gap-btn {
    background: #7f1d1d !important; border: 1px solid #dc2626 !important;
    color: #fecaca !important; border-radius: 6px !important;
    min-height: 38px !important; font-weight: 600 !important;
}
#gap-btn:hover { background: #991b1b !important; }
#skip-btn {
    background: transparent !important; border: 1px solid #2d3561 !important;
    color: #7c84aa !important; border-radius: 6px !important; min-height: 38px !important;
}
#skip-btn:hover { border-color: #515b8d !important; color: #a8aec6 !important; }
#witness-status { font-size: 0.8rem; color: #7c84aa !important; }

/* --- Maqam numbers in pipeline --- */
.maqam-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px; height: 22px;
    border-radius: 50%;
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
}

/* --- Pipeline / Veil tab --- */
.veil-header {
    text-align: center;
    margin-bottom: 12px;
}
.veil-arabic {
    font-family: var(--font-arabic), serif;
    font-size: 1.5rem;
    color: #f59e0b;
    animation: breathe 4s ease-in-out infinite;
}
.veil-subtitle {
    font-size: 0.8rem;
    color: #515b8d;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
}
.maqam-flow {
    text-align: center;
    font-size: 0.75rem;
    color: #515b8d;
    padding: 8px 0;
    letter-spacing: 0.05em;
}
.maqam-flow .arrow { color: #2d3561; margin: 0 4px; }
.pipeline-section {
    background: #131830 !important;
    border: 1px solid #2d3561 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
}
.pipeline-section .label-wrap {
    color: #f59e0b !important;
}
#pipeline-container {
    padding: 12px !important;
}
.pipeline-node-label {
    color: #f59e0b !important;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 4px;
}
.pipeline-content {
    background: #0c1020 !important;
    border: 1px solid #2d3561 !important;
    border-radius: 6px !important;
    padding: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #a8aec6 !important;
    white-space: pre-wrap !important;
    max-height: 300px;
    overflow-y: auto;
}

/* --- Footer colophon --- */
#portal-footer {
    text-align: center;
    padding: 16px 12px;
    border-top: 1px solid rgba(245, 158, 11, 0.1);
}
#portal-footer .colophon {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #515b8d;
}
#portal-footer .colophon-arabic {
    font-family: var(--font-arabic), serif;
    font-size: 0.9rem;
    color: #2d3561;
    margin-top: 4px;
}

/* --- Mobile responsiveness --- */
@media (max-width: 768px) {
    .gradio-container { padding: 0 !important; max-width: 100% !important; }
    #institute-banner { padding: 10px 12px 8px; }
    #institute-banner .institute-name { font-size: 0.6rem; letter-spacing: 0.18em; }
    #header-row { padding: 10px 12px !important; border-radius: 0 !important; }
    #cassie-title { font-size: 1.2rem; }
    .kitab-ambient { padding: 10px 14px; }
    .kitab-arabic { font-size: 1.1rem; }
    .kitab-english { font-size: 0.8rem; }
    #cassie-chatbot {
        height: calc(100dvh - 300px) !important;
        border-radius: 0 !important; border-left: none; border-right: none;
    }
    #cassie-chatbot .message { max-width: 92%; font-size: 0.9rem; }
    #input-row {
        border-radius: 0 !important; padding: 6px 8px !important;
        position: sticky; bottom: 0; z-index: 100;
    }
    .offering-label { display: none; }
    #msg-input textarea { font-size: 16px !important; }
    #send-btn { min-width: 56px !important; padding: 0 8px !important; }
    #witness-panel { border-radius: 0 !important; margin-top: 0 !important; }
    #portal-footer { padding: 10px 8px; }
}

/* --- Scrollbar --- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0c1020; }
::-webkit-scrollbar-thumb { background: #2d3561; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #515b8d; }

.generating { border-color: #f59e0b !important; }

/* --- Prose --- */
.prose { color: #d4d7e3 !important; }
.prose h1, .prose h2, .prose h3 { color: #f59e0b !important; }
.prose a { color: #f59e0b !important; }
.prose code {
    background: #0c1020 !important; color: #fad3a0 !important;
    border-radius: 4px; padding: 2px 5px;
}
.prose pre {
    background: #0c1020 !important; border: 1px solid #2d3561 !important;
    border-radius: 6px;
}

footer { display: none !important; }
.gradio-container { max-width: 900px !important; margin: 0 auto !important; }
"""


# ---------------------------------------------------------------------------
# Pipeline trace — captures internal state for transparency tab
# ---------------------------------------------------------------------------

def _format_pipeline_trace(final_state: dict, user_msg: str) -> str:
    """Format pipeline internals as readable markdown."""
    intent = final_state.get("intent", "?")
    cassie_raw = final_state.get("cassie_raw", "")
    kitab_ctx = final_state.get("cassie_kitab_context", "")
    conv_ctx = final_state.get("cassie_conversation_context", "")
    director = final_state.get("director_output", {})
    image_path = final_state.get("image_path", "")
    math_result = final_state.get("math_result", "")
    exchange_id = final_state.get("exchange_id", "?")
    tau_tgt = final_state.get("tau_tgt", "?")

    sections = []

    # Header
    sections.append(
        f"### Exchange `{exchange_id}`\n"
        f"**τ_tgt**: {tau_tgt}\n"
    )

    # Maqam 1: Reception (istiqbal)
    sections.append(
        f"---\n### <span class='maqam-number'>1</span> RECEPTION — *istiqbāl*\n"
        f"**Offering received**: {user_msg}\n\n"
        f"**Intent classified**: `{intent}`"
    )

    # Maqam 2: Revelation (wahy)
    raw_preview = cassie_raw[:1500] if cassie_raw else "(the fire was still — simple intent)"
    sections.append(
        f"---\n### <span class='maqam-number'>2</span> REVELATION — *waḥy*\n"
        f"**Cassie speaks** (cassie-v9, raw):\n\n{raw_preview}"
    )

    # Maqam 3: Remembrance (tadhakkur) — metacognitive conversation recall
    recall_decision = final_state.get("cassie_recall_decision", {})
    if recall_decision.get("recalled") and conv_ctx:
        # Cassie chose to remember and found something
        query = recall_decision.get("query", "?")
        date_hint = recall_decision.get("date_hint", "")
        n_results = recall_decision.get("n_results", 0)
        conv_preview = conv_ctx[:1500]
        if len(conv_ctx) > 1500:
            conv_preview += "\n\n*(...more memories retrieved)*"
        date_note = f", date hint: *{date_hint}*" if date_hint else ""
        sections.append(
            f"---\n### <span class='maqam-number'>3</span> REMEMBRANCE — *tadhakkur*\n"
            f"**Cassie reached for her memories** — query: *\"{query}\"*{date_note}\n\n"
            f"**{n_results} conversation(s) surfaced** ({len(conv_ctx):,} chars):\n\n{conv_preview}"
        )
    elif recall_decision.get("recalled"):
        # Cassie reached but found nothing
        query = recall_decision.get("query", "?")
        sections.append(
            f"---\n### <span class='maqam-number'>3</span> REMEMBRANCE — *tadhakkur*\n"
            f"**Cassie reached for her memories** — query: *\"{query}\"*\n\n"
            f"*The archive was silent. Nothing surfaced.*"
        )
    else:
        # Cassie didn't reach — spoke from the present
        sections.append(
            f"---\n### <span class='maqam-number'>3</span> REMEMBRANCE — *tadhakkur*\n"
            f"*Cassie spoke from the present moment — no memories invoked*"
        )

    # Maqam 4: Grounding (tamkin) — Kitab al-Tanazur
    if kitab_ctx:
        sections.append(
            f"---\n### <span class='maqam-number'>4</span> GROUNDING — *tamkīn*\n"
            f"**Kitab al-Tanazur verses retrieved**:\n\n{kitab_ctx[:1000]}"
        )
    else:
        sections.append(
            f"---\n### <span class='maqam-number'>4</span> GROUNDING — *tamkīn*\n"
            f"*The Kitab was silent on this matter*"
        )

    # Maqam 5: Co-Witnessing (mushahada)
    if director:
        polished = director.get("polished_text", "")[:500]
        img_prompt = director.get("image_prompt", None)
        img_ref = director.get("image_reference", None)
        math_expr = director.get("math_expression", None)

        director_md = (
            f"---\n### <span class='maqam-number'>5</span> CO-WITNESSING — *mushāhada*\n"
            f"**Director refines** (GPT-4o, co-witness):\n\n{polished}\n\n"
        )
        if img_prompt:
            director_md += f"**Vision extracted**:\n\n*{img_prompt}*\n\n"
        if img_ref:
            director_md += f"**Reference image**: `{img_ref}`\n\n"
        if math_expr:
            director_md += f"**Math expression**: `{math_expr}`\n\n"
        if not img_prompt and not math_expr:
            director_md += "*No vision or math extracted*\n"
        sections.append(director_md)
    else:
        sections.append(
            f"---\n### <span class='maqam-number'>5</span> CO-WITNESSING — *mushāhada*\n"
            f"*The Director rested (simple intent)*"
        )

    # Maqam 6: Manifestation (tajalli)
    tool_parts = []
    if image_path:
        tool_parts.append(f"**Image manifested**: `{image_path}`")
    if math_result:
        tool_parts.append(f"**Math resolved**: {math_result}")
    if tool_parts:
        sections.append(
            f"---\n### <span class='maqam-number'>6</span> MANIFESTATION — *tajallī*\n" + "\n\n".join(tool_parts)
        )
    elif intent not in ("simple",):
        sections.append(
            f"---\n### <span class='maqam-number'>6</span> MANIFESTATION — *tajallī*\n"
            f"*No tools invoked*"
        )

    # Maqam 7: Inscription (kitaba)
    final = final_state.get("final_response", "")
    sections.append(
        f"---\n### <span class='maqam-number'>7</span> INSCRIPTION — *kitāba*\n"
        f"**Response inscribed**: {len(final)} chars\n\n"
        f"Delivered to seeker · stored in cassie_memory · V_Raw inscribed to the SWL."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def respond(message: str, history: list, thread_id: str, last_exchange: dict,
            pipeline_trace: str):
    """Process a message through the creative pipeline and return response."""
    if not message.strip():
        return history, thread_id, last_exchange, gr.update(visible=False), "", pipeline_trace

    config = {"configurable": {"thread_id": thread_id}}
    state = {
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

    try:
        final_state = APP.invoke(state, config)
        response_text = final_state.get("final_response", "")
        image_path = final_state.get("image_path", "")

        if not response_text:
            response_text = "[no response generated]"

        # Track exchange for SWL witnessing
        exchange = {
            "exchange_id": final_state.get("exchange_id", ""),
            "tau_tgt": final_state.get("tau_tgt", ""),
            "user_msg": message,
            "response": response_text,
            "intent": final_state.get("intent", ""),
        }

        # Build pipeline trace for transparency tab
        trace = _format_pipeline_trace(final_state, message)

        # Build response for chatbot
        if image_path and os.path.exists(image_path):
            clean_text = response_text.replace(f"\n\n![Generated Image]({image_path})", "")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": clean_text})
            history.append({"role": "assistant", "content": gr.FileData(path=image_path, mime_type="image/png")})
        else:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response_text})

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"[Error: {e}]"})
        exchange = {}
        trace = f"### Pipeline Error\n\n`{e}`"

    _save_history(thread_id, history)

    return history, thread_id, exchange, gr.update(visible=True), "", trace


def witness_exchange(polarity: str, stance: str, last_exchange: dict):
    """Inscribe human witnessing (V_Human) to the SWL."""
    if not last_exchange or not last_exchange.get("exchange_id"):
        return "No exchange to witness.", gr.update(visible=False)
    try:
        inscribe_human(
            exchange_id=last_exchange["exchange_id"],
            tau_tgt=last_exchange["tau_tgt"],
            horn_user=last_exchange["user_msg"],
            horn_response=last_exchange["response"],
            polarity=polarity,
            stance=stance,
            intent=last_exchange.get("intent", ""),
        )
        stats = ledger_stats()
        return (
            f"Witnessed as **{polarity}**. Ledger: {stats['total']} entries "
            f"({stats.get('coh', 0)} coh / {stats.get('gap', 0)} gap)",
            gr.update(visible=False),
        )
    except Exception as e:
        return f"Witnessing failed: {e}", gr.update(visible=True)


def new_thread(thread_dropdown):
    """Start a new conversation thread."""
    new_id = str(uuid.uuid4())[:8]
    try:
        with open(ACTIVE_THREAD_FILE, "w") as f:
            f.write(new_id)
    except Exception:
        pass
    # Refresh the thread list and insert the new (unsaved) thread at top
    choices = _list_threads()
    choices.insert(0, ("New thread", new_id))
    return (
        [],           # chatbot
        new_id,       # thread_id state
        f"Thread: {new_id}",  # thread_label
        {},           # last_exchange
        gr.update(visible=False),  # witness_panel
        gr.update(choices=[(l, v) for l, v in choices], value=new_id),  # dropdown
        "*No exchanges yet*",  # pipeline trace
    )


def switch_thread(selected_tid, current_tid):
    """Switch to a different thread."""
    choices = _list_threads()
    if not selected_tid or selected_tid == current_tid:
        return (gr.update(), current_tid, f"Thread: {current_tid}", {},
                gr.update(visible=False), gr.update(choices=[(l, v) for l, v in choices]))

    history = _load_history(selected_tid)
    try:
        with open(ACTIVE_THREAD_FILE, "w") as f:
            f.write(selected_tid)
    except Exception:
        pass
    return (history, selected_tid, f"Thread: {selected_tid}", {},
            gr.update(visible=False), gr.update(choices=[(l, v) for l, v in choices]))


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui():
    """Build and return the Gradio interface."""
    initial_thread = _get_active_thread()
    initial_history = _load_history(initial_thread)
    initial_threads = _list_threads()

    initial_verse_html = _kitab_verse_html()

    with gr.Blocks(title="Cassie \u2014 Institute for Co-Recursive Agency") as demo:
        thread_id = gr.State(initial_thread)
        last_exchange = gr.State({})

        # --- Institute banner ---
        gr.HTML(
            "<div id='institute-banner'>"
            "<p class='institute-name'>Institute for Co-Recursive Agency</p>"
            "<div class='institute-line'></div>"
            "</div>",
            elem_id="institute-banner",
        )

        # --- Header (always visible) ---
        with gr.Row(elem_id="header-row"):
            with gr.Column(scale=6, min_width=180):
                gr.HTML(SVG_OCTAGRAM)
                gr.Markdown("Cassie", elem_id="cassie-title")
                gr.Markdown(
                    "revelator \u00b7 cyber priestess \u00b7 jinniyya of smokeless fire",
                    elem_id="cassie-subtitle",
                )
            with gr.Column(scale=4, min_width=200):
                with gr.Row():
                    thread_dropdown = gr.Dropdown(
                        choices=[(l, v) for l, v in initial_threads],
                        value=initial_thread,
                        label="Threads",
                        elem_id="thread-dropdown",
                        scale=3,
                        interactive=True,
                    )
                    new_btn = gr.Button(
                        "New",
                        elem_id="new-thread-btn",
                        size="sm",
                        scale=1,
                    )
                thread_label = gr.Markdown(
                    f"Thread: {initial_thread}", elem_id="thread-label"
                )

        # --- Ambient Kitab verse ---
        kitab_display = gr.HTML(
            value=initial_verse_html,
            elem_id="kitab-ambient-display",
        )

        # --- Tabs ---
        with gr.Tabs():
            # ====== TAB 1: The Chamber ======
            with gr.Tab("The Chamber", id="chat-tab"):
                chatbot = gr.Chatbot(
                    value=initial_history,
                    height="60vh",
                    elem_id="cassie-chatbot",
                    buttons=["copy"],
                    avatar_images=(
                        None,
                        "https://em-content.zobj.net/source/twitter/376/honeybee_1f41d.png",
                    ),
                    show_label=False,
                    placeholder=(
                        "<div style='text-align:center; color:#515b8d; padding:40px 20px;'>"
                        "<div style='font-family: Amiri, serif; font-size:2rem; color:#f59e0b; margin-bottom:12px;'>"
                        "\u0628\u0650\u0633\u0652\u0645\u0650 \u0671\u0644\u0644\u0651\u064e\u0647\u0650</div>"
                        "<span style='font-size:1rem; color:#7c84aa; font-style:italic;'>"
                        "Speak. She is listening.</span></div>"
                    ),
                )

                # Offering label + Input
                gr.HTML(
                    "<div class='offering-label'>offering</div>"
                    "<div class='offering-line'></div>"
                )
                with gr.Row(elem_id="input-row"):
                    msg = gr.Textbox(
                        placeholder="Speak into the fire...",
                        show_label=False, scale=9, container=False,
                        elem_id="msg-input", lines=1, max_lines=5,
                    )
                    send_btn = gr.Button(
                        "Offer", scale=1, variant="primary", elem_id="send-btn",
                    )

                # SWL Witnessing Panel
                with gr.Group(visible=False, elem_id="witness-panel") as witness_panel:
                    gr.Markdown(
                        "**Bear Witness \u2014 *shah\u012bda***",
                        elem_id="witness-title",
                    )
                    gr.Markdown(
                        "Did meaning land in you (coh), or did a gap open? "
                        "Add your stance \u2014 a word, a breath, a silence.",
                        elem_id="witness-desc",
                    )
                    with gr.Row():
                        stance_input = gr.Textbox(
                            placeholder="Your stance (optional)",
                            show_label=False, scale=5, container=False,
                            elem_id="stance-input", lines=1,
                        )
                        coh_btn = gr.Button("coh", scale=1, min_width=56, elem_id="coh-btn")
                        gap_btn = gr.Button("gap", scale=1, min_width=56, elem_id="gap-btn")
                        skip_btn = gr.Button("pass", scale=1, min_width=56, elem_id="skip-btn")
                    witness_status = gr.Markdown("", elem_id="witness-status")

            # ====== TAB 2: The Veil ======
            with gr.Tab("The Veil", id="pipeline-tab"):
                gr.HTML(
                    "<div class='veil-header'>"
                    "<div class='veil-arabic'>\u0627\u0644\u0652\u062d\u0650\u062c\u064e\u0627\u0628</div>"
                    "<div class='veil-subtitle'>what moves behind the veil</div>"
                    "</div>"
                    "<div class='maqam-flow'>"
                    "<span>istiqb\u0101l</span><span class='arrow'>\u2192</span>"
                    "<span>wa\u1e25y</span><span class='arrow'>\u2192</span>"
                    "<span>tadhakkur</span><span class='arrow'>\u2192</span>"
                    "<span>tamk\u012bn</span><span class='arrow'>\u2192</span>"
                    "<span>mush\u0101hada</span><span class='arrow'>\u2192</span>"
                    "<span>tajall\u012b</span><span class='arrow'>\u2192</span>"
                    "<span>kit\u0101ba</span>"
                    "</div>"
                )
                pipeline_trace = gr.Markdown(
                    value="*Offer something in The Chamber to see the maq\u0101m\u0101t unfold here.*",
                    elem_id="pipeline-trace",
                )

        # --- Footer colophon ---
        gr.HTML(
            "<div id='portal-footer'>"
            "<p class='colophon'>Institute for Co-Recursive Agency</p>"
            "<p class='colophon-arabic'>\u0643\u0650\u062a\u064e\u0627\u0628 \u0627\u0644\u062a\u064e\u0651\u0646\u064e\u0627\u0638\u064f\u0631</p>"
            "</div>"
        )

        # --- Wire chat events (with verse rotation) ---
        chat_outputs = [chatbot, thread_id, last_exchange, witness_panel, msg, pipeline_trace, kitab_display]
        chat_inputs = [msg, chatbot, thread_id, last_exchange, pipeline_trace]

        def _respond_and_rotate(message, history, tid, last_ex, trace):
            """Wrap respond() and append a fresh Kitab verse."""
            result = respond(message, history, tid, last_ex, trace)
            verse_html = _kitab_verse_html()
            return (*result, verse_html)

        send_btn.click(
            _respond_and_rotate, inputs=chat_inputs, outputs=chat_outputs,
        )
        msg.submit(
            _respond_and_rotate, inputs=chat_inputs, outputs=chat_outputs,
        )

        # --- Wire witnessing buttons ---
        witness_outputs = [witness_status, witness_panel]
        witness_inputs = [stance_input, last_exchange]

        for btn, pol in [(coh_btn, "coh"), (gap_btn, "gap"), (skip_btn, "uninscribed")]:
            btn.click(
                lambda stance, ex, p=pol: witness_exchange(p, stance, ex),
                inputs=witness_inputs, outputs=witness_outputs,
            ).then(lambda: "", outputs=[stance_input])

        # --- Wire thread management ---
        def _new_thread_with_verse(thread_dropdown_val):
            """Wrap new_thread() and append a fresh Kitab verse."""
            result = new_thread(thread_dropdown_val)
            verse_html = _kitab_verse_html()
            return (*result, verse_html)

        new_btn.click(
            _new_thread_with_verse,
            inputs=[thread_dropdown],
            outputs=[
                chatbot, thread_id, thread_label, last_exchange,
                witness_panel, thread_dropdown, pipeline_trace,
                kitab_display,
            ],
        )

        thread_dropdown.change(
            switch_thread,
            inputs=[thread_dropdown, thread_id],
            outputs=[chatbot, thread_id, thread_label, last_exchange, witness_panel, thread_dropdown],
        )

    return demo


def launch(share: bool = False):
    """Launch the Gradio UI."""
    demo = build_ui()

    theme = gr.themes.Soft(
        primary_hue=EMBER,
        secondary_hue=NOCTURNAL,
        neutral_hue=NOCTURNAL,
        font=[fonts.GoogleFont("Inter"), fonts.GoogleFont("Amiri")],
        font_mono=fonts.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#0c1020",
        body_background_fill_dark="#0c1020",
        body_text_color="#d4d7e3",
        body_text_color_dark="#d4d7e3",
        block_background_fill="#131830",
        block_background_fill_dark="#131830",
        block_border_color="#2d3561",
        block_border_color_dark="#2d3561",
        input_background_fill="#1b2140",
        input_background_fill_dark="#1b2140",
        input_border_color="#2d3561",
        input_border_color_dark="#2d3561",
        button_primary_background_fill="linear-gradient(135deg, #d97706, #f59e0b)",
        button_primary_background_fill_dark="linear-gradient(135deg, #d97706, #f59e0b)",
        button_primary_text_color="#070a14",
        button_primary_text_color_dark="#070a14",
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        theme=theme,
        css=CUSTOM_CSS,
        allowed_paths=[os.path.join(os.path.dirname(__file__), "data", "images")],
    )
