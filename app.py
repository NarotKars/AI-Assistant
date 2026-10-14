import os
import sqlite3
import hashlib
import io
import wave
import array
import math
import re
import time
from datetime import datetime
import logging
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --- Page Setup ---
st.set_page_config(
    page_title="Task Extractor",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)
logging.getLogger("google.genai").setLevel(logging.ERROR)
load_dotenv()

# --- Custom Styling ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        text-align: center;
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    [data-testid="stAudioInput"] {
        display: flex;
        justify-content: center;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- SQLite Database Setup ---
DB_NAME = "fc_tasks.db"


def get_conn():
    conn = sqlite3.connect(DB_NAME)
    # Make ON DELETE CASCADE actually work in SQLite
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            transcript TEXT,
            summary TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id INTEGER,
            title TEXT,
            assignee TEXT,
            due_date TEXT,
            priority TEXT,
            FOREIGN KEY (note_id) REFERENCES notes (id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()


init_db()


def save_note_to_db(result):
    conn = get_conn()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO notes (timestamp, transcript, summary) VALUES (?, ?, ?)",
        (timestamp, result.transcript, result.summary)
    )
    note_id = cursor.lastrowid
    for t in result.tasks:
        cursor.execute(
            "INSERT INTO tasks (note_id, title, assignee, due_date, priority) VALUES (?, ?, ?, ?, ?)",
            (note_id, t.title, t.assignee, t.due_date, t.priority)
        )
    conn.commit()
    conn.close()


def load_history_from_db():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, transcript, summary FROM notes ORDER BY id DESC")
    notes = cursor.fetchall()

    history = []
    for note in notes:
        note_id, timestamp, transcript, summary = note
        cursor.execute(
            "SELECT title, assignee, due_date, priority FROM tasks WHERE note_id = ?",
            (note_id,)
        )
        tasks_rows = cursor.fetchall()
        tasks = [
            {"title": row[0], "assignee": row[1], "due_date": row[2], "priority": row[3]}
            for row in tasks_rows
        ]
        history.append({
            "id": note_id,
            "timestamp": timestamp,
            "transcript": transcript,
            "summary": summary,
            "tasks": tasks
        })
    conn.close()
    return history


def clear_db():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks")
    cursor.execute("DELETE FROM notes")
    conn.commit()
    conn.close()


@st.cache_resource
def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


client = get_genai_client()
if client is None:
    st.error("GEMINI_API_KEY is missing from your environment variables or .env file.")
    st.stop()


class Task(BaseModel):
    title: str = Field(description="Short task summary")
    assignee: str | None = Field(description="Assigned person or None")
    due_date: str | None = Field(description="Deadline or None")
    priority: str = Field(description="High, Medium, or Low", default="Medium")


class MeetingTaskExtraction(BaseModel):
    transcript: str = Field(description="Exact transcript in original language")
    summary: str = Field(description="1-sentence summary")
    tasks: list[Task] = Field(description="Extracted tasks")


class NoSpeechError(Exception):
    """Raised when the recording contains no usable speech."""


# Tune these if real quiet speech gets rejected (lower) or noise gets through (raise)
MIN_DURATION_SEC = 1.0
MIN_RMS = 150  # 16-bit PCM scale (max 32768)


def check_audio_has_sound(audio_data: bytes):
    """Reject recordings that are too short or essentially silent (WAV only)."""
    try:
        with wave.open(io.BytesIO(audio_data), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            width = wf.getsampwidth()
            raw = wf.readframes(frames)
    except (wave.Error, EOFError):
        return  # not a parsable WAV; rely on the model-side guard instead

    if rate and frames / rate < MIN_DURATION_SEC:
        raise ValueError("Audio recording is too short.")

    if width == 2 and raw:
        samples = array.array("h")
        samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        if rms < MIN_RMS:
            raise NoSpeechError("Recording is silent.")


PROMPT = """You are a strict transcription and task-extraction assistant.

Rules:
- Use ONLY what is actually spoken in the audio. Never invent, guess, or fill in content.
- If the audio is silent, contains only background noise, or has no intelligible speech, return an empty transcript, an empty summary, and an empty tasks list.
- Only create a task if the speaker clearly states something that needs to be done. Do not infer tasks that were not said.
- Leave assignee and due_date as null unless they are explicitly mentioned.
- Keep the original language (Armenian/English). Do not translate.

Return: the exact transcript, a 1-sentence summary, and the extracted tasks."""


# Models you prefer, tried first (if your key can actually use them)
PREFERRED_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash"]
RETRIES_PER_MODEL = 2   # attempts on one model before moving to the next
MAX_MODELS = 4          # never try more than this many models per recording

# Skip models that can't take audio and return text/JSON
EXCLUDED_KEYWORDS = (
    "tts", "image", "embedding", "live", "audio", "robotics",
    "computer-use", "learnlm", "aqa", "imagen", "veo",
)


def model_sort_key(name: str):
    """Newest version first, then flash > pro > other, 'lite' last."""
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    version = float(m.group(1)) if m else 0.0
    family = 0 if "flash" in name else 1 if "pro" in name else 2
    lite = 1 if "lite" in name else 0
    return (-version, lite, family, name)


@st.cache_data(ttl=3600, show_spinner=False)
def discover_models() -> list[str]:
    """Ask the Gemini API which models this key can use for generateContent."""
    try:
        found = []
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" not in actions:
                continue
            if not name.startswith("gemini"):
                continue
            if any(k in name for k in EXCLUDED_KEYWORDS):
                continue
            found.append(name)
        return sorted(set(found), key=model_sort_key)
    except Exception:
        return []  # listing failed, fall back to the preferred list only


def get_candidate_models() -> list[str]:
    discovered = discover_models()
    if discovered:
        preferred = [m for m in PREFERRED_MODELS if m in discovered]
        rest = [m for m in discovered if m not in preferred]
        candidates = preferred + rest
    else:
        candidates = list(PREFERRED_MODELS)
    return candidates[:MAX_MODELS]


def is_transient_error(e: Exception) -> bool:
    code = getattr(e, "code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    msg = str(e)
    return any(s in msg for s in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded"))


def generate_with_retry(contents, config):
    last_error = None
    for model in get_candidate_models():
        for attempt in range(RETRIES_PER_MODEL):
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as e:
                last_error = e
                code = getattr(e, "code", None)
                if code in (401, 403):
                    raise  # key/permission problem: other models won't help
                if is_transient_error(e):
                    if attempt < RETRIES_PER_MODEL - 1:
                        time.sleep(2 ** attempt)  # 1s, then move on
                    continue
                break  # model-specific problem (unsupported, not found): next model
    raise last_error


def process_audio_note(audio_data: bytes, mime_type: str) -> MeetingTaskExtraction:
    if len(audio_data) < 1000:
        raise ValueError("Audio recording is empty or too short.")

    if "mp3" in mime_type or "mpeg" in mime_type:
        mime_type = "audio/mp3"
    else:
        mime_type = "audio/wav"
        check_audio_has_sound(audio_data)  # skip the API call for silence

    response = generate_with_retry(
        contents=[
            types.Part.from_bytes(data=audio_data, mime_type=mime_type),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MeetingTaskExtraction,
            temperature=0,
        ),
    )
    result = MeetingTaskExtraction.model_validate_json(response.text)

    # Model-side guard: nothing was actually said
    if not result.transcript.strip():
        raise NoSpeechError("No speech detected.")
    return result


# --- Sidebar: History ---
with st.sidebar:
    st.header("🗂️ History")
    history = load_history_from_db()

    if history:
        if st.button("🗑️ Clear All History", use_container_width=True):
            clear_db()
            st.rerun()
        st.divider()
        for idx, entry in enumerate(history, 1):
            with st.expander(f"[{entry['timestamp']}] {entry['summary'][:35]}..."):
                st.markdown(f"**Summary:** {entry['summary']}")
                st.markdown(f"**Tasks ({len(entry['tasks'])}):**")
                for t in entry['tasks']:
                    st.caption(f"- {t['title']} ({t['assignee'] or 'Unassigned'})")
    else:
        st.info("No saved recordings yet.")

# --- Main Interface ---
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("⚽ Hello")
st.caption("Record your briefing below")
st.markdown('</div>', unsafe_allow_html=True)

if "processed_audio_signature" not in st.session_state:
    st.session_state.processed_audio_signature = None
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "last_failed" not in st.session_state:
    st.session_state.last_failed = False


def retry_processing():
    # Forget the signature so the same recording is processed again
    st.session_state.processed_audio_signature = None
    st.session_state.last_failed = False

audio_recording = st.audio_input("Record briefing", label_visibility="collapsed")

if audio_recording is not None:
    # Identify the recording by its CONTENT (id() changes on every rerun)
    audio_data = audio_recording.getvalue()
    audio_signature = hashlib.md5(audio_data).hexdigest()

    if st.session_state.processed_audio_signature != audio_signature:
        # Mark as handled BEFORE processing so a failure or rerun can never loop
        st.session_state.processed_audio_signature = audio_signature
        try:
            with st.spinner("🎙️ Analyzing audio and extracting tasks... Please wait a few seconds."):
                mime_type = getattr(audio_recording, "type", None) or "audio/wav"
                result = process_audio_note(audio_data, mime_type)
                save_note_to_db(result)
                st.session_state.latest_result = {
                    "signature": audio_signature,
                    "summary": result.summary,
                    "tasks": [t.model_dump() for t in result.tasks],
                }
            st.rerun()
        except NoSpeechError:
            st.warning("⚠️ No speech detected in the recording. Nothing was saved.")
        except ValueError:
            st.warning("⚠️ Recording is too short.")
        except Exception as e:
            st.session_state.last_failed = True
            if is_transient_error(e):
                st.error("⚠️ The AI service is busy right now. Nothing was saved. Please try again in a moment.")
            else:
                st.error(f"Error: {e}")

    if st.session_state.last_failed:
        st.button("🔄 Try again", on_click=retry_processing)

# Show the summary and tasks only for the recording made in this session
latest = st.session_state.latest_result
if (
    audio_recording is not None
    and latest is not None
    and latest["signature"] == st.session_state.processed_audio_signature
):
    st.divider()
    st.subheader("📋 Summary")
    st.write(latest['summary'])

    st.subheader(f"✅ Extracted Tasks ({len(latest['tasks'])})")
    for idx, t in enumerate(latest['tasks'], 1):
        with st.container(border=True):
            st.markdown(f"**{idx}. {t['title']}**")
            col1, col2, col3 = st.columns(3)
            col1.caption(f"**Assignee:** {t['assignee'] or 'Unassigned'}")
            col2.caption(f"**Due:** {t['due_date'] or 'None'}")
            col3.caption(f"**Priority:** {t['priority']}")
