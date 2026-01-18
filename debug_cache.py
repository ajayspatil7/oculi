#!/usr/bin/env python3
"""
Debug cache issues
"""

import tempfile
import torch
import h5py
import numpy as np
from pathlib import Path

from oculi.cache import CaptureCache
from oculi.capture.structures import AttentionCapture, CaptureConfig

def debug_hdf5_issue():
    """Debug the HDF5 saving/loading issue."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        print(f"Using temp dir: {cache_dir}")
        
        # Create simple capture
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
            model_name="debug_model"
        )
        
        input_ids = torch.randint(0, 100, (1, 4))
        config = CaptureConfig()
        
        # Test manual HDF5 operations
        data_path = cache_dir / "debug_test.h5"
        
        print("Testing HDF5 save...")
        # Manual save
        with h5py.File(data_path, 'w') as f:
            # Save tensors
            if capture.queries is not None:
                f.create_dataset('queries', data=capture.queries.numpy(), compression='gzip')
            if capture.keys is not None:
                f.create_dataset('keys', data=capture.keys.numpy(), compression='gzip')
            if capture.values is not None:
                f.create_dataset('values', data=capture.values.numpy(), compression='gzip')
            if capture.patterns is not None:
                f.create_dataset('patterns', data=capture.patterns.numpy(), compression='gzip')
            
            # Save metadata as attributes
            f.attrs['n_layers'] = capture.n_layers
            f.attrs['n_heads'] = capture.n_heads
            f.attrs['n_kv_heads'] = capture.n_kv_heads
            f.attrs['n_tokens'] = capture.n_tokens
            f.attrs['head_dim'] = capture.head_dim
            f.attrs['model_name'] = capture.model_name.encode('utf-8')
            f.attrs['qk_stage'] = capture.qk_stage.encode('utf-8')
            f.attrs['captured_layers'] = np.array(list(capture.captured_layers), dtype=np.int32)
        
        print("Testing HDF5 load...")
        # Manual load
        with h5py.File(data_path, 'r') as f:
            print("Attributes:", dict(f.attrs))
            
            # Load tensors
            tensors = {}
            for key in ['queries', 'keys', 'values', 'patterns']:
                if key in f:
                    tensors[key] = torch.from_numpy(f[key][:])
                    print(f"Loaded {key}: {tensors[key].shape}")
                else:
                    tensors[key] = None
                    print(f"No {key} found")
            
            # Load metadata
            attrs = dict(f.attrs)
            print("Raw attrs:", attrs)
            
            # Try to access attributes
            print("n_layers:", attrs['n_layers'])
            print("model_name:", attrs['model_name'])
            
            # This is where the error likely occurs
            try:
                model_name_str = attrs['model_name']
                if isinstance(model_name_str, bytes):
                    model_name_str = model_name_str.decode('utf-8')
                elif hasattr(model_name_str, 'decode'):
                    model_name_str = model_name_str.decode('utf-8')
                else:
                    model_name_str = str(model_name_str)
                print("Decoded model_name:", model_name_str)
            except Exception as e:
                print(f"Error decoding model_name: {e}")
        
        print("Testing with CaptureCache...")
        # Now test with actual cache
        cache = CaptureCache(cache_dir, max_size_gb=0.1)
        
        try:
            cache.put(input_ids, config, capture, "debug_model")
            print("Put successful")
            
            retrieved = cache.get(input_ids, config, "debug_model")
            print(f"Get result: {retrieved}")
            
            if retrieved is not None:
                print("SUCCESS: Cache working!")
            else:
                print("FAIL: Retrieved None")
                
        except Exception as e:
            print(f"Cache error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_hdf5_issue()