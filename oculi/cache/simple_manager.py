"""
Simplified Cache Manager
========================

A cleaner implementation focusing on core caching functionality
without complex HDF5 attribute handling.
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
from typing import Optional, Dict, Any, Union, List
import torch

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    h5py = None

from oculi.capture.structures import AttentionCapture, CaptureConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Metadata for cached capture."""
    key: str
    created_time: float
    last_accessed: float
    file_size_bytes: int
    input_shape: List[int]
    config_summary: str
    model_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        return cls(**data)


class SimpleCaptureCache:
    """
    Simplified cache manager focused on core functionality.
    """
    
    def __init__(self, cache_dir: Union[str, Path], max_size_gb: float = 5.0):
        if not HDF5_AVAILABLE:
            raise ImportError("h5py required: pip install h5py")
        
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self._lock = threading.RLock()
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        (self.cache_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"Cache initialized at {self.cache_dir}")
    
    def _compute_key(self, input_ids: torch.Tensor, config: CaptureConfig, model_name: str) -> str:
        """Compute cache key."""
        input_bytes = input_ids.cpu().numpy().tobytes()
        config_dict = {
            "layers": config.layers,
            "capture_queries": config.capture_queries,
            "capture_keys": config.capture_keys,
            "capture_values": config.capture_values,
            "capture_patterns": config.capture_patterns,
            "qk_stage": config.qk_stage,
        }
        config_bytes = json.dumps(config_dict, sort_keys=True).encode()
        model_bytes = model_name.encode()
        
        combined = input_bytes + config_bytes + model_bytes
        return hashlib.sha256(combined).hexdigest()
    
    def _get_paths(self, key: str) -> tuple[Path, Path]:
        """Get metadata and data file paths."""
        meta_path = self.cache_dir / "metadata" / f"{key}.json"
        data_path = self.cache_dir / "data" / f"{key}.h5"
        return meta_path, data_path
    
    def get(self, input_ids: torch.Tensor, config: CaptureConfig, model_name: str) -> Optional[AttentionCapture]:
        """Retrieve cached capture."""
        key = self._compute_key(input_ids, config, model_name)
        meta_path, data_path = self._get_paths(key)
        
        with self._lock:
            if not (meta_path.exists() and data_path.exists()):
                return None
            
            try:
                # Update access time in metadata
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                metadata['last_accessed'] = time.time()
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                
                # Load capture from HDF5
                capture = self._load_from_hdf5(data_path)
                logger.debug(f"Cache HIT: {key[:8]}...")
                return capture
                
            except Exception as e:
                logger.warning(f"Failed to load cache {key}: {e}")
                self._remove_entry(key)
                return None
    
    def put(self, input_ids: torch.Tensor, config: CaptureConfig, capture: AttentionCapture, model_name: str) -> None:
        """Store capture in cache."""
        key = self._compute_key(input_ids, config, model_name)
        meta_path, data_path = self._get_paths(key)
        
        with self._lock:
            try:
                # Save to HDF5
                self._save_to_hdf5(capture, data_path)
                
                # Save metadata
                metadata = {
                    'key': key,
                    'created_time': time.time(),
                    'last_accessed': time.time(),
                    'file_size_bytes': data_path.stat().st_size,
                    'input_shape': list(input_ids.shape),
                    'config_summary': self._summarize_config(config),
                    'model_name': model_name
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.debug(f"Cached: {key[:8]}... ({metadata['file_size_bytes'] / 1024 / 1024:.1f} MB)")
                
                # Enforce size limit
                self._enforce_size_limit()
                
            except Exception as e:
                logger.error(f"Failed to cache {key}: {e}")
                self._remove_entry(key)
                raise
    
    @contextmanager
    def cached_capture(self, adapter, input_ids: torch.Tensor, config: Optional[CaptureConfig] = None):
        """Context manager for automatic caching."""
        if config is None:
            config = CaptureConfig()
        
        # Try cache first
        cached = self.get(input_ids, config, adapter.model_name)
        if cached is not None:
            yield cached
            return
        
        # Compute and cache
        logger.debug("Cache MISS: computing capture...")
        capture = adapter.capture(input_ids, config)
        
        try:
            self.put(input_ids, config, capture, adapter.model_name)
        except Exception as e:
            logger.warning(f"Failed to cache: {e}")
        
        yield capture
    
    def _load_from_hdf5(self, data_path: Path) -> AttentionCapture:
        """Load capture from HDF5 file."""
        with h5py.File(data_path, 'r') as f:
            # Load tensors
            tensors = {}
            for key in ['queries', 'keys', 'values', 'patterns']:
                if key in f:
                    tensors[key] = torch.from_numpy(f[key][:])
                else:
                    tensors[key] = None
            
            # Load metadata from attributes
            attrs = dict(f.attrs)
            
            # Handle attribute conversion robustly
            def convert_attr(attr_value):
                # Handle numpy arrays/scalars
                if hasattr(attr_value, 'shape'):
                    if attr_value.shape == ():
                        # 0-dimensional array (scalar)
                        return attr_value.item()
                    else:
                        # Multi-dimensional array
                        return attr_value.tolist()
                elif isinstance(attr_value, bytes):
                    return attr_value.decode('utf-8')
                else:
                    return attr_value
            
            # Convert all attributes
            converted_attrs = {k: convert_attr(v) for k, v in attrs.items()}
            
            return AttentionCapture(
                queries=tensors['queries'],
                keys=tensors['keys'],
                values=tensors['values'],
                patterns=tensors['patterns'],
                n_layers=int(converted_attrs['n_layers']),
                n_heads=int(converted_attrs['n_heads']),
                n_kv_heads=int(converted_attrs['n_kv_heads']),
                n_tokens=int(converted_attrs['n_tokens']),
                head_dim=int(converted_attrs['head_dim']),
                model_name=str(converted_attrs['model_name']),
                qk_stage=str(converted_attrs['qk_stage']),
                captured_layers=tuple(int(x) for x in converted_attrs['captured_layers'])
            )
    
    def _save_to_hdf5(self, capture: AttentionCapture, data_path: Path) -> None:
        """Save capture to HDF5 file."""
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
            
            # Save metadata as attributes
            f.attrs['n_layers'] = capture.n_layers
            f.attrs['n_heads'] = capture.n_heads
            f.attrs['n_kv_heads'] = capture.n_kv_heads
            f.attrs['n_tokens'] = capture.n_tokens
            f.attrs['head_dim'] = capture.head_dim
            f.attrs['model_name'] = capture.model_name.encode('utf-8')
            f.attrs['qk_stage'] = capture.qk_stage.encode('utf-8')
            f.attrs['captured_layers'] = list(capture.captured_layers)
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry."""
        meta_path, data_path = self._get_paths(key)
        try:
            meta_path.unlink(missing_ok=True)
            data_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to remove {key}: {e}")
    
    def _enforce_size_limit(self) -> None:
        """Remove oldest entries if needed."""
        # Get all entries with access times
        entries = []
        for meta_file in (self.cache_dir / "metadata").glob("*.json"):
            try:
                with open(meta_file, 'r') as f:
                    data = json.load(f)
                entries.append((data['last_accessed'], data['key'], meta_file))
            except Exception:
                meta_file.unlink(missing_ok=True)
        
        # Sort by access time (oldest first)
        entries.sort()
        
        # Remove until under limit
        total_size = sum(
            (self.cache_dir / "data" / f"{key}.h5").stat().st_size
            for _, key, _ in entries
            if (self.cache_dir / "data" / f"{key}.h5").exists()
        )
        
        for access_time, key, meta_file in entries:
            if total_size <= self.max_size_bytes:
                break
                
            data_path = self.cache_dir / "data" / f"{key}.h5"
            if data_path.exists():
                size = data_path.stat().st_size
                self._remove_entry(key)
                total_size -= size
                logger.debug(f"Evicted {key[:8]}... ({size / 1024 / 1024:.1f} MB)")
    
    def _summarize_config(self, config: CaptureConfig) -> str:
        """Summarize config for metadata."""
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
            }
    
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


# Export the simplified version
CaptureCache = SimpleCaptureCache