# Installation

## Requirements

Oculi requires:

- **Python:** 3.10 or higher
- **PyTorch:** 2.0.0 or higher
- **Transformers:** 4.30.0 or higher

## Installation Methods

### From PyPI (Recommended)

!!! note "Coming Soon"
    PyPI distribution will be available with v1.0.0 release.

### From Source

Install directly from GitHub:

```bash
# Clone the repository
git clone https://github.com/ajayspatil7/oculi.git
cd oculi

# Install in editable mode
pip install -e .
```

### Optional Dependencies

Install additional features as needed:

```bash
# Visualization support
pip install -e ".[viz]"

# Development tools (testing, linting)
pip install -e ".[dev]"

# Documentation tools
pip install -e ".[docs]"

# Everything
pip install -e ".[all]"
```

## Verify Installation

Verify your installation:

```python
import oculi
print(f"Oculi version: {oculi.__version__}")

# Test basic import
from oculi.models.llama import LlamaAttentionAdapter
from oculi.analysis import AttributionMethods, CompositionAnalysis
print("✓ All imports successful!")
```

## Platform-Specific Notes

### macOS (Apple Silicon)

For Apple Silicon Macs (M1/M2/M3/M4):

```bash
# Use MPS backend for acceleration
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Oculi works on CPU, but some operations may be slower. For best performance, use CUDA-enabled systems for large models.

### Linux (CUDA)

Ensure you have CUDA-compatible PyTorch:

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Windows

Oculi supports Windows with standard Python installation. Use WSL2 for best compatibility with CUDA.

## Testing Without GPU

Oculi includes mock models for testing on CPU:

```python
from tests.mocks import MockLlamaAdapter

# Tiny mock model for testing
adapter = MockLlamaAdapter()
input_ids = adapter.tokenize("Test input")

# All features work
capture = adapter.capture(input_ids)
print(f"✓ Capture successful: {capture.patterns.shape}")
```

## Troubleshooting

### ImportError: No module named 'oculi'

Ensure you installed in editable mode with `-e` flag:

```bash
pip install -e .
```

### CUDA out of memory

For large models, use selective capture:

```python
from oculi import CaptureConfig

config = CaptureConfig(
    layers=[20, 21, 22],  # Only specific layers
    capture_values=False   # Skip values if not needed
)

capture = adapter.capture(input_ids, config=config)
```

### Version conflicts

Check your environment:

```bash
pip list | grep -E "torch|transformers|oculi"
```

Ensure compatible versions:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0

## Next Steps

- [Quick Start Guide](quick-start.md) - Get started in 5 minutes
- [Core Concepts](core-concepts.md) - Understand key abstractions
- [User Guides](../guides/attention-capture.md) - In-depth feature documentation
