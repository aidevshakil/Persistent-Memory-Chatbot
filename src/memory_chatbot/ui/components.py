"""Reusable Streamlit fragments: memory chips, the sidebar and the header.

Kept free of business logic so the page module reads as a straight sequence of
"render this, then that".
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Sequence

import streamlit as st

from memory_chatbot.chat.engine import ChatEngine
from memory_chatbot.models import (
    MemoryCategory,
    MemoryRecord,
    MemorySearchResult,
    MemoryWrite,
)

CATEGORY_ICONS: dict[MemoryCategory, str] = {
    MemoryCategory.IDENTITY: "🪪",
    MemoryCategory.LOCATION: "📍",
    MemoryCategory.PREFERENCE: "❤️",
    MemoryCategory.INTEREST: "🎯",
    MemoryCategory.OCCUPATION: "💼",
    MemoryCategory.GOAL: "🚀",
    MemoryCategory.HEALTH: "🥗",
    MemoryCategory.RELATIONSHIP: "👥",
    MemoryCategory.SKILL: "🛠️",
    MemoryCategory.CONSTRAINT: "⛔",
    MemoryCategory.OTHER: "🗒️",
}


def render_header(engine: ChatEngine, user_id: str, fact_count: int) -> None:
    """Title plus a one-line "what is actually running" status."""
    st.title("🧠 Persistent Memory Chatbot")
    st.markdown(
        '<p class="status-line">'
        f"LLM <code>{html.escape(engine.provider_name)}</code> · "
        f"model <code>{html.escape(engine.model_name)}</code> · "
        f"memory <code>{html.escape(engine.backend_name)}</code> · "
        f"user <code>{html.escape(user_id)}</code> · "
        f"{fact_count} facts stored"
        "</p>",
        unsafe_allow_html=True,
    )
    if engine.is_offline:
        st.warning(
            "No LLM API key found, so replies come from the built-in **offline echo** "
            "provider. Memory extraction still runs on deterministic rules, so you can "
            "verify persistence end-to-end. Add `GROQ_API_KEY` (or `GOOGLE_API_KEY` / "
            "`OPENAI_API_KEY`) to `.env` for real conversation.",
            icon="🔌",
        )


def render_memory_chips(results: Sequence[MemorySearchResult]) -> None:
    """Show which memories were injected into a turn, with their scores."""
    if not results:
        return
    chips = "".join(
        '<span class="mem-chip">'
        f"{CATEGORY_ICONS.get(item.record.category, '🗒️')} "
        f"{html.escape(item.record.as_sentence())}"
        f'<span class="mem-score"> · {item.score:.2f}</span>'
        "</span>"
        for item in results
    )
    with st.expander(f"🧠 {len(results)} memories used for this reply", expanded=False):
        st.markdown(chips, unsafe_allow_html=True)
        st.caption("Score blends semantic/lexical match, importance and recency.")


def render_write_summary(writes: Sequence[MemoryWrite]) -> None:
    """Report what the turn changed in the store."""
    persisted = [w for w in writes if w.operation.value != "noop"]
    if not persisted:
        return
    lines = [f"- `{w.operation.value}` {w.describe()[2:]}" for w in persisted]
    st.success("**Memory updated**\n" + "\n".join(lines), icon="💾")


def render_stored_facts(
    records: Sequence[MemoryRecord], *, on_delete: Callable[[str], None]
) -> None:
    """Sidebar listing of every stored fact, grouped by category."""
    if not records:
        st.caption("Nothing stored yet. Say something like *“I'm 25 and I love football.”*")
        return

    grouped: dict[MemoryCategory, list[MemoryRecord]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(record)

    for category in sorted(grouped, key=str):
        icon = CATEGORY_ICONS.get(category, "🗒️")
        st.markdown(f'<div class="mem-cat">{icon} {category}</div>', unsafe_allow_html=True)
        for record in grouped[category]:
            row, button = st.columns([0.85, 0.15], vertical_alignment="center")
            with row:
                old = (
                    f'<span class="mem-old"> (was: {html.escape(record.history[0])})</span>'
                    if record.history
                    else ""
                )
                st.markdown(
                    f'<div class="mem-fact"><b>{html.escape(record.key.replace("_", " "))}</b>: '
                    f"{html.escape(record.value)}{old}</div>",
                    unsafe_allow_html=True,
                )
            with button:
                st.button(
                    "✕",
                    key=f"forget-{record.id}",
                    help=(
                        f"Forget this · importance {record.importance:.2f} · "
                        f"updated {record.updated_at:%Y-%m-%d %H:%M} UTC"
                    ),
                    on_click=on_delete,
                    args=(record.id,),
                )


def render_add_memory_form(engine: ChatEngine, user_id: str) -> None:
    """Manual fact entry - handy for seeding a demo or fixing a bad extraction."""
    with st.form("add-memory", clear_on_submit=True, border=False):
        category = st.selectbox(
            "Category",
            list(MemoryCategory),
            index=list(MemoryCategory).index(MemoryCategory.PREFERENCE),
            format_func=lambda c: f"{CATEGORY_ICONS.get(c, '🗒️')} {c}",
        )
        key = st.text_input("Key", placeholder="favourite_sport")
        value = st.text_input("Value", placeholder="football")
        importance = st.slider("Importance", 0.0, 1.0, 0.7, 0.05)
        if st.form_submit_button("Store fact", use_container_width=True):
            if not key.strip() or not value.strip():
                st.warning("Both key and value are required.")
            else:
                write = engine.memory.remember_fact(
                    user_id,
                    category=category,
                    key=key,
                    value=value,
                    importance=importance,
                )
                st.toast(f"Stored: {write.describe()}", icon="💾")
                st.rerun()


def render_export_button(records: Sequence[MemoryRecord], user_id: str) -> None:
    """Download the raw memory rows - proof they are real, portable data."""
    payload = [record.to_dict() for record in records]
    st.download_button(
        f"⬇️ Export {len(payload)} memories (JSON)",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name=f"memories-{user_id}.json",
        mime="application/json",
        use_container_width=True,
        disabled=not payload,
    )
