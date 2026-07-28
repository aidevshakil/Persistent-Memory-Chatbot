"""Streamlit entry point.

    streamlit run app.py

Kept deliberately thin: it only puts ``src/`` on the path (so the project runs
without an editable install) and hands over to the UI package.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory_chatbot.logging_utils import configure_logging  # noqa: E402
from memory_chatbot.ui import run  # noqa: E402

configure_logging()
run()
