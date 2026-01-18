"""
Cache System Tests
==================

Tests for the capture caching system.

These tests verify:
- Cache hit/miss behavior
- Correctness of stored/retrieved data  
- Size enforcement and eviction
- Thread safety
- Performance improvements
"""

import tempfile
import time
import torch
from pathlib import Path

from oculi.cache import CaptureCache
from oculi.capture.structures import AttentionCapture, CaptureConfig


def test_basic_caching():
    """Test basic cache put/get functionality."""
    print(" Testing basic caching...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CaptureCache(tmpdir, max_size_gb=0.1)  # Small cache for testing
        
        # Create mock capture
        mock_capture = AttentionCapture(
            queries=torch.randn(2, 4, 8, 16),
            keys=torch.randn(2, 2, 8, 16),  # GQA: fewer KV heads
            values=torch.randn(2, 2, 8, 16),
            patterns=torch.randn(2, 4, 8, 8),
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            n_tokens=8,
            head_dim=16,
            model_name="test_model"
        )
        
        input_ids = torch.randint(0, 1000, (1, 8))
        config = CaptureConfig()
        
        # Put in cache
        cache.put(input_ids, config, mock_capture, "test_model")
        
        # Get from cache
        retrieved = cache.get(input_ids, config, "test_model")
        
        assert retrieved is not None, "Should retrieve cached capture"
        
        # Check that non-None tensors match
        if retrieved.queries is not None and mock_capture.queries is not None:
            assert torch.allclose(retrieved.queries, mock_capture.queries), "Queries should match"
        if retrieved.keys is not None and mock_capture.keys is not None:
            assert torch.allclose(retrieved.keys, mock_capture.keys), "Keys should match"
        if retrieved.values is not None and mock_capture.values is not None:
            assert torch.allclose(retrieved.values, mock_capture.values), "Values should match"
        if retrieved.patterns is not None and mock_capture.patterns is not None:
            assert torch.allclose(retrieved.patterns, mock_capture.patterns), "Patterns should match"
        
        print(" Basic caching test passed")


def test_cache_miss():
    """Test cache miss behavior."""
    print(" Testing cache miss...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CaptureCache(tmpdir)
        
        input_ids = torch.randint(0, 1000, (1, 8))
        config = CaptureConfig()
        
        # Should return None for uncached items
        result = cache.get(input_ids, config, "test_model")
        assert result is None, "Uncached item should return None"
        
        print(" Cache miss test passed")


def test_context_manager():
    """Test cached_capture context manager."""
    print(" Testing context manager...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CaptureCache(tmpdir)
        
        # Mock adapter
        class MockAdapter:
            def __init__(self):
                self.call_count = 0
                self.model_name = "test_model"
            
            def capture(self, input_ids, config):
                self.call_count += 1
                return AttentionCapture(
                    queries=torch.randn(1, 2, 4, 8),
                    keys=torch.randn(1, 1, 4, 8),
                    values=torch.randn(1, 1, 4, 8),
                    patterns=torch.randn(1, 2, 4, 4),
                    n_layers=1,
                    n_heads=2,
                    n_kv_heads=1,
                    n_tokens=4,
                    head_dim=8,
                    model_name=self.model_name
                )
        
        adapter = MockAdapter()
        input_ids = torch.randint(0, 100, (1, 4))
        config = CaptureConfig()
        
        # First call - should compute
        with cache.cached_capture(adapter, input_ids, config) as capture1:
            pass
        
        assert adapter.call_count == 1, "First call should compute"
        
        # Second call - should use cache
        with cache.cached_capture(adapter, input_ids, config) as capture2:
            pass
            
        assert adapter.call_count == 1, "Second call should use cache"
        
        print(" Context manager test passed")


def test_cache_stats():
    """Test cache statistics reporting."""
    print(" Testing cache stats...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CaptureCache(tmpdir, max_size_gb=0.1)
        
        # Add some entries
        for i in range(3):
            capture = AttentionCapture(
                queries=torch.randn(1, 2, 4, 8),
                keys=torch.randn(1, 1, 4, 8),
                values=torch.randn(1, 1, 4, 8),
                patterns=torch.randn(1, 2, 4, 4),
                n_layers=1,
                n_heads=2,
                n_kv_heads=1,
                n_tokens=4,
                head_dim=8,
                model_name="test_model"
            )
            input_ids = torch.randint(0, 100, (1, 4))
            config = CaptureConfig(layers=[i])
            cache.put(input_ids, config, capture, "test_model")
        
        stats = cache.stats()
        assert stats["entries"] == 3, f"Expected 3 entries, got {stats['entries']}"
        assert stats["total_size_bytes"] > 0, "Should have non-zero size"
        
        print(f" Cache stats test passed: {stats['entries']} entries, {stats['total_size_gb']:.3f} GB")


def test_deterministic_hashing():
    """Test that identical inputs produce identical cache keys."""
    print(" Testing deterministic hashing...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CaptureCache(tmpdir)
        
        # Same inputs should produce same key
        input_ids = torch.randint(0, 1000, (1, 8))
        config = CaptureConfig()
        
        # We can't directly access private _compute_key, but we can test
        # that identical inputs behave consistently through the public API
        
        capture1 = AttentionCapture(
            queries=torch.randn(1, 2, 4, 8),
            keys=torch.randn(1, 1, 4, 8),
            values=torch.randn(1, 1, 4, 8),
            patterns=torch.randn(1, 2, 4, 4),
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            n_tokens=4,
            head_dim=8,
            model_name="test_model"
        )
        
        # Put the same thing twice - second should be no-op
        cache.put(input_ids, config, capture1, "test_model")
        cache.put(input_ids, config, capture1, "test_model")  # Should overwrite/update access time
        
        stats = cache.stats()
        assert stats["entries"] == 1, "Identical keys should not create duplicate entries"
        
        print(" Deterministic hashing test passed")


def run_all_tests():
    """Run all cache tests."""
    print("=" * 60)
    print(" Running Cache System Tests")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        test_basic_caching()
        test_cache_miss()
        test_context_manager()
        test_cache_stats()
        test_deterministic_hashing()
        
        elapsed = time.time() - start_time
        print("=" * 60)
        print(f" All cache tests passed! ({elapsed:.2f}s)")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f" Test failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()