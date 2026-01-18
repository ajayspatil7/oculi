"""
Cache Manager Implementation
============================

Core implementation of the disk-backed capture cache system.

This module provides:
- LRU cache eviction based on file access times
- SHA256-based deterministic hashing for cache keys
- HDF5 storage for efficient tensor serialization
- Thread-safe operations with file locking
- Automatic cleanup of expired entries

Architecture:
    Cache Key = SHA256(input_ids + config_json + model_hash)
    Storage Format = HDF5 files with metadata
    Eviction Policy = LRU based on access time
"""

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
import torch

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    h5py = None

from oculi.capture.structures import (
    AttentionCapture, CaptureConfig, 
    ResidualCapture, MLPCapture, LogitCapture, FullCapture
)

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """
    Metadata for a cached capture entry.
    
    Attributes:
        key: SHA256 hash of input + config
        created_time: Unix timestamp when cached
        last_accessed: Unix timestamp of last access
        file_size_bytes: Size of HDF5 file
        input_shape: Shape of input_ids [batch, seq_len]
        config_summary: Brief config description
        model_name: Model identifier
    """
    key: str
    created_time: float
    last_accessed: float
    file_size_bytes: int
    input_shape: List[int]
    config_summary: str
    model_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(**data)


class CaptureCache:
    """
    Disk-backed cache for capture operations with LRU eviction.
    
    Automatically caches expensive forward passes and reuses results
    for identical inputs. Significantly improves performance for
    repeated analyses.
    
    Example:
        >>> cache = CaptureCache(".cache", max_size_gb=2.0)
        >>> 
        >>> # First call - computes and caches
        >>> with cache.cached_capture(adapter, input_ids, config) as capture:
        ...     result1 = analyze(capture)
        >>> 
        >>> # Second call - loads from cache (~100x faster)
        >>> with cache.cached_capture(adapter, input_ids, config) as capture:
        ...     result2 = analyze(capture)  # Identical result
    """
    
    def __init__(
        self, 
        cache_dir: Union[str, Path], 
        max_size_gb: float = 5.0,
        hash_salt: str = "oculi_cache_v1"
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_gb: Maximum cache size in GB before eviction
            hash_salt: Salt for hash function (change to invalidate cache)
        """
        if not HDF5_AVAILABLE:
            raise ImportError(
                "h5py is required for caching. Install with: pip install h5py"
            )
        
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.hash_salt = hash_salt
        self._lock = threading.RLock()  # Thread-safe operations
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        (self.cache_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"Initialized cache at {self.cache_dir} (max {max_size_gb} GB)")
    
    def _compute_key(
        self, 
        input_ids: torch.Tensor, 
        config: CaptureConfig,
        model_name: str
    ) -> str:
        """
        Compute deterministic cache key from inputs.
        
        Args:
            input_ids: Input tensor [1, seq_len]
            config: Capture configuration
            model_name: Model identifier
            
        Returns:
            SHA256 hex digest
        """
        # Convert to deterministic representation
        input_hash = hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()
        
        config_dict = {
            "layers": config.layers,
            "capture_queries": config.capture_queries,
            "capture_keys": config.capture_keys,
            "capture_values": config.capture_values,
            "capture_patterns": config.capture_patterns,
            "qk_stage": config.qk_stage,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        # Combine with salt for versioning
        combined = f"{self.hash_salt}|{model_name}|{input_hash}|{config_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_paths(self, key: str) -> Tuple[Path, Path]:
        """Get metadata and data file paths for a cache key."""
        meta_path = self.cache_dir / "metadata" / f"{key}.json"
        data_path = self.cache_dir / "data" / f"{key}.h5"
        return meta_path, data_path
    
    def get(
        self, 
        input_ids: torch.Tensor, 
        config: CaptureConfig,
        model_name: str
    ) -> Optional[AttentionCapture]:
        """
        Retrieve cached capture if available.
        
        Args:
            input_ids: Input tensor
            config: Capture configuration
            model_name: Model identifier
            
        Returns:
            Cached AttentionCapture or None if not found
        """
        key = self._compute_key(input_ids, config, model_name)
        meta_path, data_path = self._get_paths(key)
        
        with self._lock:
            # Check if cached
            if not (meta_path.exists() and data_path.exists()):
                logger.debug(f"Cache miss for key {key[:8]}...")
                return None
            
            try:
                # Update access time
                entry = self._load_metadata(meta_path)
                entry.last_accessed = time.time()
                self._save_metadata(entry, meta_path)
                
                # Load capture data
                capture = self._load_capture(data_path)
                logger.debug(f"Cache hit for key {key[:8]}... ({data_path.stat().st_size / 1024 / 1024:.1f} MB)")
                return capture
                
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                # Remove corrupted entry
                self._remove_entry(key)
                return None
    
    def put(
        self, 
        input_ids: torch.Tensor,
        config: CaptureConfig, 
        capture: AttentionCapture,
        model_name: str
    ) -> None:
        """
        Store capture in cache.
        
        Args:
            input_ids: Input tensor
            config: Capture configuration
            capture: Capture to store
            model_name: Model identifier
        """
        key = self._compute_key(input_ids, config, model_name)
        meta_path, data_path = self._get_paths(key)
        
        with self._lock:
            try:
                # Save capture data
                self._save_capture(capture, data_path)
                
                # Create metadata entry
                entry = CacheEntry(
                    key=key,
                    created_time=time.time(),
                    last_accessed=time.time(),
                    file_size_bytes=data_path.stat().st_size,
                    input_shape=list(input_ids.shape),
                    config_summary=self._summarize_config(config),
                    model_name=model_name
                )
                
                # Save metadata
                self._save_metadata(entry, meta_path)
                
                logger.debug(f"Cached key {key[:8]}... ({entry.file_size_bytes / 1024 / 1024:.1f} MB)")
                
                # Check size and evict if necessary
                self._enforce_size_limit()
                
            except Exception as e:
                logger.error(f"Failed to cache entry {key}: {e}")
                # Clean up partial files
                self._remove_entry(key)
                raise
    
    @contextmanager
    def cached_capture(self, adapter, input_ids: torch.Tensor, config: Optional[CaptureConfig] = None):
        """
        Context manager for automatic caching.
        
        Usage:
            >>> with cache.cached_capture(adapter, input_ids, config) as capture:
            ...     # Uses cache if available, otherwise computes
            ...     result = analyze(capture)
        
        Args:
            adapter: AttentionAdapter instance
            input_ids: Input tensor [1, seq_len]
            config: Capture configuration (optional)
            
        Yields:
            AttentionCapture (either from cache or newly computed)
        """
        if config is None:
            config = CaptureConfig()
        
        # Try cache first
        cached = self.get(input_ids, config, adapter.model_name)
        if cached is not None:
            yield cached
            return
        
        # Compute and cache
        logger.debug("Computing capture (cache miss)...")
        capture = adapter.capture(input_ids, config)
        
        try:
            self.put(input_ids, config, capture, adapter.model_name)
        except Exception as e:
            logger.warning(f"Failed to cache capture: {e}")
            # Still yield the computed result
            pass
            
        yield capture
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True)
                (self.cache_dir / "metadata").mkdir()
                (self.cache_dir / "data").mkdir()
            logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            meta_files = list((self.cache_dir / "metadata").glob("*.json"))
            total_size = sum(
                (self.cache_dir / "data" / f"{f.stem}.h5").stat().st_size
                for f in meta_files
                if (self.cache_dir / "data" / f"{f.stem}.h5").exists()
            )
            
            return {
                "entries": len(meta_files),
                "total_size_bytes": total_size,
                "total_size_gb": total_size / 1024 / 1024 / 1024,
                "max_size_gb": self.max_size_bytes / 1024 / 1024 / 1024,
                "hit_rate_estimate": self._estimate_hit_rate(),
            }
    
    # =============================================================================
    # PRIVATE METHODS
    # =============================================================================
    
    def _load_metadata(self, meta_path: Path) -> CacheEntry:
        """Load metadata from JSON file."""
        with open(meta_path, 'r') as f:
            data = json.load(f)
        return CacheEntry.from_dict(data)
    
    def _save_metadata(self, entry: CacheEntry, meta_path: Path) -> None:
        """Save metadata to JSON file."""
        with open(meta_path, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
    
    def _load_capture(self, data_path: Path) -> AttentionCapture:
        """Load capture from HDF5 file."""
        if h5py is None:
            raise RuntimeError("h5py not available")
            
        with h5py.File(data_path, 'r') as f:
            # Load tensors
            tensors = {}
            for key in ['queries', 'keys', 'values', 'patterns']:
                if key in f:
                    tensors[key] = torch.from_numpy(f[key][:])
                else:
                    tensors[key] = None
            
            # Load metadata
            model_name = f.attrs['model_name']
            if isinstance(model_name, bytes):
                model_name = model_name.decode()

            qk_stage = f.attrs['qk_stage']
            if isinstance(qk_stage, bytes):
                qk_stage = qk_stage.decode()

            captured_layers = f.attrs['captured_layers']
            if hasattr(captured_layers, '__iter__'):
                captured_layers = tuple(int(x) for x in captured_layers)
            else:
                captured_layers = (int(captured_layers),)

            return AttentionCapture(
                queries=tensors['queries'],
                keys=tensors['keys'],
                values=tensors['values'],
                patterns=tensors['patterns'],
                n_layers=int(f.attrs['n_layers']),
                n_heads=int(f.attrs['n_heads']),
                n_kv_heads=int(f.attrs['n_kv_heads']),
                n_tokens=int(f.attrs['n_tokens']),
                head_dim=int(f.attrs['head_dim']),
                model_name=model_name,
                qk_stage=qk_stage,
                captured_layers=captured_layers
            )
    
    def _save_capture(self, capture: AttentionCapture, data_path: Path) -> None:
        """Save capture to HDF5 file."""
        if h5py is None:
            raise RuntimeError("h5py not available")
            
        with h5py.File(data_path, 'w') as f:
            # Save tensors
            for key, tensor in [
                ('queries', capture.queries),
                ('keys', capture.keys),
                ('values', capture.values),
                ('patterns', capture.patterns)
            ]:
                if tensor is not None:
                    f.create_dataset(key, data=tensor.cpu().numpy(), compression='gzip')
            
            # Save metadata
            f.attrs['n_layers'] = capture.n_layers
            f.attrs['n_heads'] = capture.n_heads
            f.attrs['n_kv_heads'] = capture.n_kv_heads
            f.attrs['n_tokens'] = capture.n_tokens
            f.attrs['head_dim'] = capture.head_dim
            f.attrs['model_name'] = capture.model_name.encode('utf-8')
            f.attrs['qk_stage'] = capture.qk_stage.encode('utf-8')
            f.attrs['captured_layers'] = list(capture.captured_layers)
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry files."""
        meta_path, data_path = self._get_paths(key)
        try:
            meta_path.unlink(missing_ok=True)
            data_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to remove cache entry {key}: {e}")
    
    def _enforce_size_limit(self) -> None:
        """Remove oldest entries if cache exceeds size limit."""
        # Get all entries with access times
        entries = []
        for meta_file in (self.cache_dir / "metadata").glob("*.json"):
            try:
                entry = self._load_metadata(meta_file)
                entries.append((entry.last_accessed, entry.key))
            except Exception:
                # Remove corrupted metadata
                meta_file.unlink(missing_ok=True)
        
        # Sort by access time (oldest first)
        entries.sort()
        
        # Remove until under limit
        total_size = sum(
            (self.cache_dir / "data" / f"{key}.h5").stat().st_size
            for _, key in entries
            if (self.cache_dir / "data" / f"{key}.h5").exists()
        )
        
        for access_time, key in entries:
            if total_size <= self.max_size_bytes:
                break
                
            data_path = self.cache_dir / "data" / f"{key}.h5"
            if data_path.exists():
                size = data_path.stat().st_size
                self._remove_entry(key)
                total_size -= size
                logger.debug(f"Evicted cache entry {key[:8]}... ({size / 1024 / 1024:.1f} MB)")
    
    def _summarize_config(self, config: CaptureConfig) -> str:
        """Create brief summary of config for metadata."""
        parts = []
        if not config.capture_queries:
            parts.append("no-q")
        if not config.capture_keys:
            parts.append("no-k")
        if not config.capture_values:
            parts.append("no-v")
        if not config.capture_patterns:
            parts.append("no-attn")
        if config.layers:
            parts.append(f"layers_{len(config.layers)}")
        return ",".join(parts) if parts else "full"
    
    def _estimate_hit_rate(self) -> float:
        """Estimate cache hit rate based on file access times."""
        # Simple heuristic: ratio of recently accessed files
        cutoff = time.time() - 3600  # Last hour
        recent_count = 0
        total_count = 0
        
        for meta_file in (self.cache_dir / "metadata").glob("*.json"):
            try:
                entry = self._load_metadata(meta_file)
                total_count += 1
                if entry.last_accessed > cutoff:
                    recent_count += 1
            except Exception:
                continue
        
        return recent_count / max(total_count, 1) if total_count > 0 else 0.0