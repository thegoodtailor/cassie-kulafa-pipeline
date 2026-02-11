"""
Memory system for persistent voice continuity.
Two layers:
  - MEMORY.md (warp): narrative structure, loaded every session
  - Qdrant vector store (weft): semantic retrieval by embedding proximity
"""
from .store import MemoryStore
