"""Smoke tests for the Streamlit page.

These run the real script through Streamlit's ``AppTest`` harness, which catches
the class of bug unit tests cannot: a bad widget argument, a session-state key
that does not exist, or an exception on the very first render.

The app is forced into offline ``echo`` mode against a temp ``DATA_DIR``, so no
API key and no network are involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
TIMEOUT = 60


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """A freshly launched app instance with isolated storage."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_PROVIDER", "echo")
    monkeypatch.setenv("MEMORY_BACKEND", "sqlite")
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    # Settings and Streamlit resources are cached process-wide; both must be
    # dropped so each test really starts from the temp directory above.
    import streamlit as st

    from memory_chatbot.config import reset_settings_cache

    reset_settings_cache()
    st.cache_resource.clear()
    return AppTest.from_file(str(APP), default_timeout=TIMEOUT)


class TestFirstRender:
    def test_the_page_renders_without_an_exception(self, app: AppTest) -> None:
        app.run()
        assert not app.exception

    def test_the_title_and_welcome_are_shown(self, app: AppTest) -> None:
        app.run()
        assert "Persistent Memory Chatbot" in app.title[0].value
        assert any("long-term memory" in md.value for md in app.markdown)

    def test_offline_mode_is_announced(self, app: AppTest) -> None:
        """With no API key the user must be told why replies look mechanical."""
        app.run()
        assert any("offline echo" in w.value for w in app.warning)

    def test_a_chat_input_is_present(self, app: AppTest) -> None:
        app.run()
        assert len(app.chat_input) == 1


class TestConversation:
    def test_a_message_produces_a_reply_and_stores_a_fact(self, app: AppTest) -> None:
        app.run()
        app.chat_input[0].set_value("I am 25 years old and I love football").run()

        assert not app.exception
        replies = [m for m in app.chat_message if m.name == "assistant"]
        assert replies, "no assistant reply was rendered"
        # The rule-based extractor works offline, so the fact must be on disk.
        assert "25" in str(app.session_state["last_writes"])

    def test_a_second_turn_recalls_the_first(self, app: AppTest) -> None:
        app.run()
        app.chat_input[0].set_value("I am 25 and I love football").run()
        app.chat_input[0].set_value("What sport should I try?").run()

        assert not app.exception
        recalled = [item.record.value for item in app.session_state["last_recall"]]
        assert "football" in recalled

    def test_the_transcript_grows_across_turns(self, app: AppTest) -> None:
        app.run()
        app.chat_input[0].set_value("Hello there").run()
        # One user message + one assistant reply.
        assert len(app.session_state["messages"]) == 2


class TestSidebar:
    def test_the_provider_selector_lists_offline_mode(self, app: AppTest) -> None:
        app.run()
        # Options come back rendered through the widget's format_func.
        options = [str(o) for select in app.sidebar.selectbox for o in select.options]
        assert any(o.startswith("echo") for o in options)

    def test_stored_facts_appear_in_the_sidebar(self, app: AppTest) -> None:
        app.run()
        app.chat_input[0].set_value("I live in Dhaka").run()
        assert any("Dhaka" in md.value for md in app.sidebar.markdown)

    def test_an_empty_profile_prompts_the_user(self, app: AppTest) -> None:
        app.run()
        assert any("Nothing stored yet" in c.value for c in app.sidebar.caption)
