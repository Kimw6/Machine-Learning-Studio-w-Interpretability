from __future__ import annotations
import streamlit as st
from typing import Dict
from datetime import datetime

# Canonical steps (order matters)
STEPS: Dict[str, str] = {
    "qs_upload":    "Upload/Load Data",
    "qs_target":    "Select Target",
    "qs_eda":       "Run EDA",
    "qs_train":     "Train Model",
    "qs_interpret": "Interpret Results",
}

# ---------- State helpers ----------
def _init():
    """Initialize progress + timestamp state once per session."""
    if "qs" not in st.session_state:
        st.session_state.qs = {k: False for k in STEPS.keys()}
    if "qs_ts" not in st.session_state:
        st.session_state.qs_ts = {k: None for k in STEPS.keys()}

def _now_str() -> str:
    # Keep format short + readable; server local time
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def set_done(step_key: str, done: bool = True):
    """Mark a step done/undone. Set or clear timestamp accordingly."""
    _init()
    if step_key not in st.session_state.qs:
        return
    prev = st.session_state.qs[step_key]
    st.session_state.qs[step_key] = done
    if done and not prev:
        st.session_state.qs_ts[step_key] = _now_str()
    if not done:
        st.session_state.qs_ts[step_key] = None

def is_done(step_key: str) -> bool:
    _init()
    return bool(st.session_state.qs.get(step_key, False))

def progress_ratio() -> float:
    _init()
    total = len(st.session_state.qs)
    done = sum(1 for v in st.session_state.qs.values() if v)
    return done / total if total else 0.0

def reset():
    """Reset all steps + timestamps."""
    st.session_state.qs  = {k: False for k in STEPS.keys()}
    st.session_state.qs_ts = {k: None for k in STEPS.keys()}

# ---------- UI ----------
def render_sidebar(read_only: bool = True):
    """
    Render the persistent checklist in the sidebar.
    - read_only=True makes checkboxes disabled (not clickable).
    - Shows live progress and timestamps.
    - Includes a 'Reset progress' control with confirmation.
    """
    _init()
    with st.sidebar:
        st.subheader("⚡ Quick Start Progress")
        st.progress(progress_ratio())
        st.caption(_summary_text())

        # Read-only checklist (disabled checkboxes)
        for k, label in STEPS.items():
            ts = st.session_state.qs_ts.get(k)
            suffix = f"  \n<span style='font-size:12px;color:gray'>✓ {ts}</span>" if ts else ""
            # Disabled checkbox mirrors current state (no user edits)
            st.checkbox(label, value=st.session_state.qs[k], disabled=True, key=f"ro_{k}")
            if ts:
                st.markdown(suffix, unsafe_allow_html=True)

        if st.session_state.get("qs_reset_done"):
            st.session_state.pop("qs_reset_confirm", None)  # safely remove it
            st.session_state["qs_reset_done"] = False


def _summary_text() -> str:
    done = sum(1 for v in st.session_state.qs.values() if v)
    total = len(st.session_state.qs)
    return f"{done}/{total} steps completed"
