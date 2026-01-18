"""
Caching System for Oculi
========================

Provides disk-backed caching for expensive capture operations.

The cache system enables:
- Reusing captures for identical inputs (huge performance gain)
- Offloading memory pressure by storing tensors on disk
- Persistent storage across sessions
- Automatic cleanup with LRU eviction

Usage:
    >>> from oculi.cache import CaptureCache
    >>> 
    >>> cache = CaptureCache(".cache", max_size_gb=5.0)
    >>> 
    >>> # Auto-caching context manager
    >>> with cache.cached_capture(adapter, input_ids, config) as capture:
    ...     # First call: computes and caches
    ...     # Subsequent calls: loads from cache
    ...     analyze(capture)

Key Components:
    - CaptureCache: Main cache manager
    - CacheEntry: Cached item metadata
    - Hash utilities: Deterministic input hashing
"""

from .manager import CaptureCache, CacheEntry

__all__ = ["CaptureCache", "CacheEntry"]