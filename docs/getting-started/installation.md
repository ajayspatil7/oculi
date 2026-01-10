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

### macOS (Apple Silicon) - MPS Support ✨

Oculi **fully supports** Apple Silicon (M1/M2/M3/M4) with MPS (Metal Performance Shaders) acceleration!

**Requirements:**
- PyTorch 2.0.0+ (for optimal MPS support)
- macOS 12.3+

**Setup:**

```bash
# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Usage:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from oculi.models.llama import LlamaAttentionAdapter
from oculi.utils import get_default_device

# Auto-detect best device (MPS on Apple Silicon)
device = get_default_device()
print(f"Using device: {device}")  # mps

# Load model on MPS
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,  # Recommended for MPS
    device_map="mps"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Create adapter - works seamlessly on MPS
adapter = LlamaAttentionAdapter(model, tokenizer)
```

**Performance Tips:**
- Use `torch.float16` for better memory efficiency on MPS
- Smaller models (1B-8B) work well on Apple Silicon
- For models >8B, monitor system memory usage

**Device Detection:**

Oculi automatically detects and uses the best available device:

```python
from oculi.utils import get_device_info

info = get_device_info()
print(info)
# DeviceInfo(device_type='mps', device=device(type='mps'), device_name='Apple Silicon (MPS)')
```

### Linux (CUDA)

Ensure you have CUDA-compatible PyTorch:

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify CUDA device
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**Usage:**

```python
from oculi.utils import get_default_device

device = get_default_device()  # Automatically selects CUDA if available
print(f"Using: {device}")  # cuda:0
```

### Windows

Oculi supports Windows with standard Python installation:

- **With NVIDIA GPU:** Use CUDA (see Linux section)
- **Without GPU:** CPU mode works perfectly
- **WSL2:** Recommended for best CUDA compatibility

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
