"""
MPS (Apple Silicon) Device Example
===================================

This example demonstrates how to use Oculi with Apple Silicon's MPS backend
for GPU acceleration on M1/M2/M3/M4 Macs.

Requirements:
- PyTorch 2.0+ with MPS support
- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 12.3+

Setup:
    export PYTORCH_ENABLE_MPS_FALLBACK=1
"""

import torch
from oculi.utils import (
    get_default_device,
    get_device_info,
    is_mps_available,
    auto_select_device,
)


def main():
    print("=" * 60)
    print("Oculi MPS (Apple Silicon) Device Example")
    print("=" * 60)

    # 1. Check MPS availability
    print("\n1. Device Availability Check")
    print("-" * 60)
    print(f"MPS Available: {is_mps_available()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"PyTorch Version: {torch.__version__}")

    # 2. Get comprehensive device info
    print("\n2. Device Information")
    print("-" * 60)
    device_info = get_device_info()
    print(device_info)
    print(f"  Device Type: {device_info.device_type}")
    print(f"  Device Name: {device_info.device_name}")
    print(f"  Has MPS: {device_info.has_mps}")
    print(f"  Has CUDA: {device_info.has_cuda}")

    # 3. Auto-select device (prefers CUDA > MPS > CPU)
    print("\n3. Auto-Select Device")
    print("-" * 60)
    device = get_default_device()
    print(f"Selected device: {device}")

    # 4. Prefer MPS over CUDA (useful on Apple Silicon)
    print("\n4. Prefer MPS Device")
    print("-" * 60)
    mps_device = auto_select_device(prefer_cuda=False)
    print(f"Selected device (MPS preferred): {mps_device}")

    # 5. Test basic tensor operations on MPS
    if is_mps_available():
        print("\n5. Testing MPS Operations")
        print("-" * 60)

        # Create tensors on MPS
        device = torch.device("mps")
        print(f"Creating tensors on {device}...")

        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)

        print(f"Tensor A: shape={a.shape}, device={a.device}")
        print(f"Tensor B: shape={b.shape}, device={b.device}")

        # Matrix multiplication
        print("Performing matrix multiplication...")
        c = torch.matmul(a, b)
        print(f"Result: shape={c.shape}, device={c.device}")

        # Attention-like operations
        print("Simulating attention computation...")
        q = torch.randn(1, 8, 512, 64, device=device)
        k = torch.randn(1, 8, 512, 64, device=device)

        scores = torch.matmul(q, k.transpose(-2, -1)) / 8.0
        attn = torch.softmax(scores, dim=-1)
        print(f"Attention: shape={attn.shape}, device={attn.device}")

        # Transfer to CPU
        print("Transferring result to CPU...")
        attn_cpu = attn.cpu()
        print(f"CPU Attention: shape={attn_cpu.shape}, device={attn_cpu.device}")

        print(" All MPS operations successful!")
    else:
        print("\n5. MPS not available - skipping MPS tests")

    # 6. Usage with Oculi adapter
    print("\n6. Using with Oculi Adapter")
    print("-" * 60)
    print("To use Oculi with MPS:")
    print("""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from oculi.models.llama import LlamaAttentionAdapter
    from oculi.utils import get_default_device

    # Auto-detect device
    device = get_default_device()

    # Load model on MPS
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16,  # Recommended for MPS
        device_map="mps"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Create adapter - automatically uses MPS
    adapter = LlamaAttentionAdapter(model, tokenizer)

    # Capture works seamlessly on MPS
    input_ids = tokenizer.encode("Hello world", return_tensors="pt").to(device)
    capture = adapter.capture(input_ids)
    """)

    # 7. Performance tips
    print("\n7. MPS Performance Tips")
    print("-" * 60)
    print("""
    1. Use torch.float16 for better memory efficiency:
       model = model.half()

    2. Enable MPS fallback for unsupported ops:
       export PYTORCH_ENABLE_MPS_FALLBACK=1

    3. Smaller models (1B-8B) work best on Apple Silicon

    4. Monitor system memory (MPS shares with system):
       - Activity Monitor > Memory tab

    5. Close other memory-intensive apps when running large models

    6. Use selective capture for memory efficiency:
       config = CaptureConfig(layers=[20, 21, 22])
    """)

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
