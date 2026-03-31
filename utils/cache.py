"""Lightweight LRU-cache utilities for the EEG Analysis Platform.

Provides a ``smart_cache`` decorator that caches return values of expensive
functions keyed on all positional and keyword arguments.  The cache is stored
as a module-level dict, so it persists across Streamlit reruns within the same
Python process.

Usage
-----
>>> from utils.cache import smart_cache
>>>
>>> @smart_cache(max_size=4)
... def expensive_computation(raw, param):
...     ...
"""

import functools
from collections import OrderedDict
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class _LRUCache:
    """A simple, thread-safe LRU cache backed by an :class:`OrderedDict`."""

    def __init__(self, max_size: int = 8) -> None:
        self._max_size = max(1, max_size)
        self._cache: OrderedDict = OrderedDict()

    def get(self, key: Any) -> Any:
        """Return the cached value, or raise ``KeyError`` if not present."""
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: Any, value: Any) -> None:
        """Store *value* under *key*, evicting the LRU entry if needed."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


def smart_cache(max_size: int = 8) -> Callable[[F], F]:
    """Decorator factory that attaches an LRU cache to a function.

    The cache key is built from all positional and keyword arguments.
    Arguments must be hashable.  MNE ``Raw`` objects are keyed by their
    ``id()`` (memory address) since they are mutable and not hashable.

    Parameters
    ----------
    max_size : int
        Maximum number of results to keep in the cache.

    Returns
    -------
    decorator : Callable
        A decorator that wraps the target function with caching logic.

    Examples
    --------
    >>> @smart_cache(max_size=3)
    ... def run_ica(raw, n_components=20):
    ...     ...
    """
    def decorator(func: F) -> F:
        cache = _LRUCache(max_size=max_size)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build a hashable cache key
            key_parts = []
            for a in args:
                try:
                    hash(a)
                    key_parts.append(a)
                except TypeError:
                    key_parts.append(id(a))
            for k, v in sorted(kwargs.items()):
                try:
                    hash(v)
                    key_parts.append((k, v))
                except TypeError:
                    key_parts.append((k, id(v)))
            key = (func.__qualname__, tuple(key_parts))

            if key in cache:
                return cache.get(key)

            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        # Expose cache management on the wrapper
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        wrapper.cache_info = lambda: {"size": len(cache), "max_size": max_size}  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
