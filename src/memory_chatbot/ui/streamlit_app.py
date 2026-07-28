"""``StreamlitChatUI`` - the chat front end.

Streamlit re-runs this module top to bottom on every interaction, so the two
rules that shape the code are:

* expensive objects (LLM client, DB connections) live in ``st.cache_resource``
  and are keyed by the settings that built them;
* everything the user should still see after a rerun lives in ``st.session_state``.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from memory_chatbot.chat.engine import DEFAULT_USER_ID, ChatEngine, build_chat_engine
from memory_chatbot.chat.transcript import ConversationStore
from memory_chatbot.config import LLMProvider, MemoryBackendName, Settings, get_settings
from memory_chatbot.logging_utils import get_logger
from memory_chatbot.models import ChatMessage, MemoryRecord, MemorySearchResult, Role
from memory_chatbot.ui.components import (
    render_add_memory_form,
    render_export_button,
    render_header,
    render_memory_chips,
    render_stored_facts,
    render_write_summary,
)
from memory_chatbot.ui.styles import CUSTOM_CSS

logger = get_logger(__name__)

WELCOME = (
    "Hi! I keep a long-term memory of what you tell me, so anything you share "
    "now will still be here after you close the app.\n\n"
    "Try: **“I'm 25, I live in Dhaka and I love football.”** Then restart the app "
    "and ask **“What sport should I try this weekend?”**"
)

_SUGGESTIONS = (
    "I'm 25 and I live in Dhaka",
    "I love football and spicy food",
    "What sport should I try this weekend?",
    "What do you remember about me?",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _cached_engine(provider: str, backend: str, model: str) -> ChatEngine:
    """One engine per (provider, backend, model) combination.

    The cache key is explicit rather than derived from ``Settings`` so switching
    provider in the sidebar reuses an already-built engine instead of
    reconnecting. Engines built for different providers open their own connection
    to the same database file, which SQLite in WAL mode handles safely.
    """
    settings = get_settings().model_copy(
        update={
            "llm_provider": LLMProvider(provider),
            "memory_backend": MemoryBackendName(backend),
            "llm_model": model,
        }
    )
    return build_chat_engine(settings)


@st.cache_resource(show_spinner=False)
def _cached_transcript(db_path: str) -> ConversationStore:
    return ConversationStore(db_path)


def _engine_for(settings: Settings, provider: LLMProvider) -> ChatEngine:
    """Engine for the currently selected provider, sharing the memory backend."""
    from memory_chatbot.config import DEFAULT_MODELS

    model = settings.llm_model if provider is settings.llm_provider else DEFAULT_MODELS[provider]
    return _cached_engine(str(provider), str(settings.memory_backend), model)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_state(settings: Settings) -> None:
    st.session_state.setdefault("user_id", DEFAULT_USER_ID)
    st.session_state.setdefault("provider", settings.llm_provider)
    st.session_state.setdefault("messages", None)
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("last_recall", [])
    st.session_state.setdefault("last_writes", [])


def _load_history(transcript: ConversationStore, user_id: str) -> list[ChatMessage]:
    """Restore the visible conversation for *user_id* from disk."""
    return transcript.load(user_id, limit=200)


def _messages() -> list[ChatMessage]:
    return st.session_state["messages"] or []


def _switch_user(transcript: ConversationStore, user_id: str) -> None:
    st.session_state["user_id"] = user_id
    st.session_state["messages"] = _load_history(transcript, user_id)
    st.session_state["last_recall"] = []
    st.session_state["last_writes"] = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar(
    engine: ChatEngine,
    settings: Settings,
    transcript: ConversationStore,
    records: Sequence[MemoryRecord],
) -> None:
    user_id: str = st.session_state["user_id"]

    with st.sidebar:
        st.subheader("👤 Session")
        known = sorted({user_id, DEFAULT_USER_ID, *engine.memory.known_users(), *transcript.users()})
        chosen = st.selectbox(
            "Memory profile",
            known,
            index=known.index(user_id),
            help="Each profile has a completely separate long-term memory.",
        )
        if chosen != user_id:
            _switch_user(transcript, chosen)
            st.rerun()

        with st.popover("➕ New profile", use_container_width=True):
            new_user = st.text_input("Profile id", placeholder="shakil", key="new-user-input")
            if st.button("Create & switch", use_container_width=True):
                cleaned = new_user.strip().replace(" ", "_")
                if cleaned:
                    _switch_user(transcript, cleaned)
                    st.rerun()

        # -------------------------------------------------------- provider
        st.subheader("⚙️ Model")
        providers = settings.available_providers()
        current = st.session_state["provider"]
        if current not in providers:
            current = providers[0]
        picked = st.selectbox(
            "LLM provider",
            providers,
            index=providers.index(current),
            format_func=lambda p: f"{p} (offline demo)" if p is LLMProvider.ECHO else str(p),
            help="Only providers with an API key in .env are listed.",
        )
        if picked != st.session_state["provider"]:
            st.session_state["provider"] = picked
            st.rerun()
        st.caption(f"Model: `{engine.model_name}`")

        # ---------------------------------------------------------- memory
        st.divider()
        st.subheader(f"🧠 Long-term memory ({len(records)})")
        st.caption(f"`{engine.backend_name}` → `{engine.memory.storage_location}`")

        def _forget(record_id: str) -> None:
            engine.memory.forget(record_id)
            st.toast("Memory deleted", icon="🗑️")

        render_stored_facts(records, on_delete=_forget)

        st.divider()
        with st.expander("➕ Add a fact manually"):
            render_add_memory_form(engine, user_id)

        if records and st.button("🪄 Summarise this profile", use_container_width=True):
            with st.spinner("Summarising…"):
                st.info(engine.profile_summary(user_id))

        render_export_button(records, user_id)

        # ------------------------------------------------------- danger zone
        st.divider()
        with st.expander("🧹 Reset"):
            st.caption("Clearing the chat keeps your long-term memories.")
            if st.button("Clear chat only", use_container_width=True):
                transcript.clear(user_id)
                st.session_state["messages"] = []
                st.session_state["last_recall"] = []
                st.session_state["last_writes"] = []
                st.rerun()
            if st.button("⚠️ Forget everything", use_container_width=True, type="primary"):
                removed = engine.memory.forget_all(user_id)
                transcript.clear(user_id)
                st.session_state["messages"] = []
                st.toast(f"Deleted {removed} memories for “{user_id}”", icon="🧹")
                st.rerun()


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


def _render_history(messages: Sequence[ChatMessage]) -> None:
    for message in messages:
        if message.role is Role.SYSTEM:
            continue
        with st.chat_message(str(message.role)):
            st.markdown(message.content)


def _render_suggestions() -> None:
    """Starter prompts, shown only on an empty conversation."""
    st.markdown("**Try one of these:**")
    for column, text in zip(st.columns(len(_SUGGESTIONS)), _SUGGESTIONS, strict=True):
        with column:
            if st.button(text, use_container_width=True, key=f"suggest-{text}"):
                st.session_state["pending_prompt"] = text
                st.rerun()


def _handle_turn(
    engine: ChatEngine, transcript: ConversationStore, user_id: str, prompt: str
) -> None:
    """Recall -> stream the reply -> persist facts, updating the UI as we go."""
    user_message = ChatMessage(role=Role.USER, content=prompt)
    st.session_state["messages"] = [*_messages(), user_message]
    transcript.append(user_id, user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    history = _messages()[:-1]  # exclude the message we are answering
    with st.spinner("Recalling what I know about you…"):
        recalled: list[MemorySearchResult] = engine.recall(user_id, prompt)

    with st.chat_message("assistant"):
        render_memory_chips(recalled)
        reply_text = st.write_stream(
            engine.stream_response(user_id, prompt, history=history, memories=recalled)
        )
    reply_text = reply_text if isinstance(reply_text, str) else "".join(reply_text)

    reply = ChatMessage(
        role=Role.ASSISTANT,
        content=reply_text,
        used_memory_ids=[item.record.id for item in recalled],
    )
    st.session_state["messages"] = [*_messages(), reply]
    transcript.append(user_id, reply)

    with st.spinner("Updating memory…"):
        writes = engine.remember(user_id, prompt, reply_text)

    st.session_state["last_recall"] = recalled
    st.session_state["last_writes"] = writes
    if any(w.operation.value != "noop" for w in writes):
        st.rerun()  # refresh the sidebar fact list


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Render the whole app. Called by ``app.py``."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    _init_state(settings)
    transcript = _cached_transcript(str(settings.sqlite_path))
    engine = _engine_for(settings, st.session_state["provider"])
    user_id: str = st.session_state["user_id"]

    if st.session_state["messages"] is None:
        st.session_state["messages"] = _load_history(transcript, user_id)

    # Fetched once and shared: the header, the sidebar list and the export
    # button all describe the same set of facts.
    records = engine.memory.profile(user_id)
    render_header(engine, user_id, len(records))
    _render_sidebar(engine, settings, transcript, records)

    messages = _messages()
    if not messages:
        with st.chat_message("assistant"):
            st.markdown(WELCOME)
        _render_suggestions()
    else:
        _render_history(messages)
        render_write_summary(st.session_state["last_writes"])

    prompt = st.chat_input("Tell me something about yourself, or ask me anything…")
    if pending := st.session_state.pop("pending_prompt", None):
        prompt = pending
    if prompt:
        _handle_turn(engine, transcript, user_id, prompt)


__all__ = ["run"]
