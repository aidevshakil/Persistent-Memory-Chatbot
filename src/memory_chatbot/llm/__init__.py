"""LLM abstraction layer.

``base``        provider contract + JSON repair
``providers``   Groq, Gemini, OpenAI, offline Echo
``embeddings``  optional embedders + vector helpers
``connector``   :class:`~memory_chatbot.llm.connector.LLMConnector`

Import from the modules directly; this package intentionally re-exports nothing,
so there is no second list of names to keep in sync.
"""
