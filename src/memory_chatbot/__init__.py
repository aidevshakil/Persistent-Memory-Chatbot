"""Persistent Memory Chatbot.

A Streamlit chatbot that remembers facts about its user across sessions.

Layers
------
``config``  environment-driven settings
``models``  domain objects shared by every layer
``llm``     pluggable LLM providers + :class:`~memory_chatbot.llm.LLMConnector`
``memory``  pluggable memory backends + :class:`~memory_chatbot.memory.MemoryHandler`
``chat``    :class:`~memory_chatbot.chat.ChatEngine`, wires memory into the LLM
``ui``      Streamlit front end
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
