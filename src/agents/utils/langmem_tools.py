from __future__ import annotations

"""
LangMem memory integration helpers.

This module wires up LangMem's hot-path tools and an in-memory store
so they can be attached to existing agents without changing current tools.

Design goals
- Use external `langmem` APIs for tool creation.
- Default to `InMemoryStore` and a single namespace for now.
- Keep an easy path to switch storage backend (e.g., Postgres) later.

Exports
- get_inmemory_store: Build a LangGraph InMemoryStore with embedding index.
- create_search_memory_tool: Proxy to langmem's tool (namespace-aware).
- create_manage_memory_tool: Proxy to langmem's tool (namespace-aware).
- create_memory_store_manager: Proxy to langmem background manager, with
  a store defaulting to in-memory. Signature differences are handled
  defensively to stay compatible across langmem versions.
"""

from typing import Iterable, Tuple

from langgraph.store.memory import InMemoryStore

# Import external langmem APIs (tools + background manager)
try:
    from langmem import (
        create_search_memory_tool as _create_search_memory_tool,
        create_manage_memory_tool as _create_manage_memory_tool,
        create_memory_store_manager as _create_memory_store_manager,
    )
except Exception as e:  # pragma: no cover - surface a clear import-time error
    raise ImportError(
        "langmem is required. Ensure `langmem>=0.0.29` is installed."
    ) from e


DEFAULT_NAMESPACE: Tuple[str, ...] = ("memories",)


def get_inmemory_store(
    *,
    dims: int = 384,
    embed: str = "huggingface:sentence-transformers/all-MiniLM-L6-v2",
):
    """Create an in-process store with a vector index.

    Notes
    - This uses HuggingFace `sentence-transformers/all-MiniLM-L6-v2` by default
      (384 dims). You can switch to another embedding model by changing
      `embed` and `dims`.
    - For production persistence, replace with a DB-backed store later.
    """
    return InMemoryStore(index={"dims": dims, "embed": embed})


def create_search_memory_tool(
    *,
    namespace: Iterable[str] = DEFAULT_NAMESPACE,
):
    """Create LangMem's search tool (hot path) bound to a namespace.

    Delegates to `langmem.create_search_memory_tool`.
    """
    return _create_search_memory_tool(namespace=tuple(namespace))


def create_manage_memory_tool(
    *,
    namespace: Iterable[str] = DEFAULT_NAMESPACE,
):
    """Create LangMem's manage tool (hot path) bound to a namespace.

    Delegates to `langmem.create_manage_memory_tool`.
    """
    return _create_manage_memory_tool(namespace=tuple(namespace))


def create_memory_store_manager(
    *,
    store=None,
    namespace: Iterable[str] = DEFAULT_NAMESPACE,
    **kwargs,
):
    """Create the background memory store manager with an in-memory store.

    Behavior
    - If `store` is not provided, builds an `InMemoryStore` using
      `get_inmemory_store()`.
    - Calls through to `langmem.create_memory_store_manager`, passing
      `store` and `namespace` where supported. Different langmem versions
      may accept different signatures; this function handles the common
      permutations defensively.
    """
    if store is None:
        store = get_inmemory_store()

    ns = tuple(namespace)

    # Try the most explicit signature first
    try:
        return _create_memory_store_manager(store=store, namespace=ns, **kwargs)
    except TypeError:
        pass

    # Some versions might not take `store` or `namespace` explicitly
    try:
        return _create_memory_store_manager(namespace=ns, **kwargs)
    except TypeError:
        pass

    try:
        return _create_memory_store_manager(store=store, **kwargs)
    except TypeError:
        # Fall back to bare call if API changed; surface error to user then
        return _create_memory_store_manager(**kwargs)


__all__ = [
    "get_inmemory_store",
    "create_search_memory_tool",
    "create_manage_memory_tool",
    "create_memory_store_manager",
]
