"""
Smart caching system for EEG Analysis Platform.

Provides LRU caching with automatic invalidation for computationally
expensive operations like ICA and PSD analysis.
"""

from functools import wraps, lru_cache
from typing import Any, Callable, Dict, Tuple
import hashlib
import pickle
import numpy as np
import mne

from utils.logger import get_logger

logger = get_logger("cache")


class SmartCache:
    """
    Smart cache with dependency tracking and automatic invalidation.

    This cache system detects when input parameters change and automatically
    invalidates cached results to prevent stale computations.
    """

    def __init__(self, max_size: int = 3):
        """
        Initialize SmartCache.

        Parameters
        ----------
        max_size : int
            Maximum number of cached items per function.
        """
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, str]] = {}

    def _compute_hash(self, obj: Any) -> str:
        """
        Compute hash of an object for caching.

        Parameters
        ----------
        obj : Any
            Object to hash.

        Returns
        -------
        str
            MD5 hash of the object.
        """
        try:
            if isinstance(obj, mne.io.Raw):
                # For MNE objects, use data hash
                data_hash = hashlib.md5(
                    obj.get_data().tobytes()
                ).hexdigest()
                return data_hash
            elif isinstance(obj, np.ndarray):
                return hashlib.md5(obj.tobytes()).hexdigest()
            else:
                return hashlib.md5(pickle.dumps(obj)).hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute hash: {str(e)}")
            return str(hash(obj))

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to add caching to a function.

        Parameters
        ----------
        func : Callable
            Function to cache.

        Returns
        -------
        Callable
            Wrapped function with caching.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            args_hash = "-".join(
                [self._compute_hash(arg) for arg in args]
            )
            kwargs_hash = "-".join(
                [f"{k}:{self._compute_hash(v)}" for k, v in kwargs.items()]
            )
            cache_key = f"{func.__name__}:{args_hash}:{kwargs_hash}"

            # Check cache
            if cache_key in self.cache:
                result, stored_hash = self.cache[cache_key]
                logger.debug(f"Cache hit for {func.__name__}")
                return result

            # Compute result
            logger.debug(f"Computing {func.__name__} (cache miss)")
            result = func(*args, **kwargs)

            # Store in cache
            if len(self.cache) >= self.max_size:
                # Remove oldest item
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Evicted {oldest_key} from cache")

            self.cache[cache_key] = (result, args_hash)
            return result

        return wrapper


def smart_cache(max_size: int = 3) -> Callable:
    """
    Decorator for smart caching with automatic invalidation.

    Parameters
    ----------
    max_size : int
        Maximum number of cached items.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @smart_cache(max_size=3)
    >>> def expensive_function(data):
    ...     return result
    """
    cache = SmartCache(max_size=max_size)
    return cache


# Standard LRU cache wrapper for simple cases
def cached(maxsize: int = 128) -> Callable:
    """
    Simple LRU cache decorator.

    Parameters
    ----------
    maxsize : int
        Maximum cache size.

    Returns
    -------
    Callable
        Decorator function.
    """
    return lru_cache(maxsize=maxsize)