#!/usr/bin/env python3
"""
Test simplified cache implementation
"""

import tempfile
import torch
from pathlib import Path
import sys

# Add oculi to path
sys.path.insert(0, str(Path(__file__).parent))

from oculi.cache.simple_manager import SimpleCaptureCache
from oculi.capture.structures import AttentionCapture, CaptureConfig

def test_simple_cache():
    """Test the simplified cache implementation."""
    
    print(" Testing simplified cache...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SimpleCaptureCache(tmpdir, max_size_gb=0.1)
        
        # Create test capture
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
        config = CaptureConfig()
        
        print("Testing PUT operation...")
        # Test put
        cache.put(input_ids, config, capture, "test_model")
        print(" PUT successful")
        
        print("Testing GET operation...")
        # Test get
        retrieved = cache.get(input_ids, config, "test_model")
        
        if retrieved is not None:
            print(" GET successful")
            print(f"  Retrieved shapes:")
            print(f"    queries: {retrieved.queries.shape if retrieved.queries is not None else 'None'}")
            print(f"    keys: {retrieved.keys.shape if retrieved.keys is not None else 'None'}")
            print(f"    values: {retrieved.values.shape if retrieved.values is not None else 'None'}")
            print(f"    patterns: {retrieved.patterns.shape if retrieved.patterns is not None else 'None'}")
        else:
            print(" GET failed - returned None")
            return False
        
        print("Testing cache stats...")
        stats = cache.stats()
        print(f" Stats: {stats['entries']} entries, {stats['total_size_gb']:.3f} GB")
        
        return True

def test_context_manager():
    """Test context manager functionality."""
    
    print("\n Testing context manager...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SimpleCaptureCache(tmpdir)
        
        # Mock adapter
        class MockAdapter:
            def __init__(self):
                self.call_count = 0
                self.model_name = "mock_model"
            
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
        
        print("First call (should compute)...")
        with cache.cached_capture(adapter, input_ids, config) as capture1:
            pass
        print(f" Adapter calls: {adapter.call_count}")
        
        print("Second call (should use cache)...")
        with cache.cached_capture(adapter, input_ids, config) as capture2:
            pass
        print(f" Adapter calls: {adapter.call_count}")
        
        # Should still be 1 if caching works
        success = adapter.call_count == 1
        print(f"{'' if success else ''} Caching {'works' if success else 'failed'}")
        
        return success

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Simplified Cache Implementation")
    print("=" * 50)
    
    try:
        test1_result = test_simple_cache()
        test2_result = test_context_manager()
        
        print("=" * 50)
        if test1_result and test2_result:
            print(" All tests PASSED!")
        else:
            print(" Some tests FAILED!")
        print("=" * 50)
        
    except Exception as e:
        print(f" Test error: {e}")
        import traceback
        traceback.print_exc()