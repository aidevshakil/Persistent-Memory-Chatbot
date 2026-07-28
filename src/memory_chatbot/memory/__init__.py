"""Memory layer.

``base``            the :class:`~memory_chatbot.memory.base.BaseMemoryBackend` contract
``sqlite_backend``  default store: one SQLite file, no services
``mem0_backend``    Mem0 OSS store
``scoring``         pure relevance-ranking functions
``extraction``      fact extraction + conflict resolution
``handler``         :class:`~memory_chatbot.memory.handler.MemoryHandler`, the policy facade
``registry``        backend selection with graceful fallback

Import from the modules directly; this package intentionally re-exports nothing.
"""
