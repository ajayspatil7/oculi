# LLaMA Attention Architecture Notes

This document explains how LLaMA implements attention. Read this when debugging or understanding capture results.

## Module Structure

```
model.model.layers[i]
├── self_attn
│   ├── q_proj: Linear(4096 → 4096)    # 32 heads × 128 dim
│   ├── k_proj: Linear(4096 → 1024)    # 8 heads × 128 dim (GQA!)
│   ├── v_proj: Linear(4096 → 1024)    # 8 heads × 128 dim
│   └── o_proj: Linear(4096 → 4096)
├── mlp
│   ├── gate_proj
│   ├── up_proj
│   └── down_proj
├── input_layernorm
└── post_attention_layernorm
```

## GQA (Grouped Query Attention)

LLaMA-3-8B uses **32 query heads** but only **8 KV heads**.

```
Query Heads:   [0] [1] [2] [3]   [4] [5] [6] [7]   ...   [28] [29] [30] [31]
                    ↓                 ↓                        ↓
KV Heads:          [0]               [1]            ...       [7]
```

Each KV head is shared by 4 query heads. This reduces memory for KV cache by 4×.

**Before attention computation:**

```python
# K, V have shape [batch, 8, seq, 128]
# Need to expand to [batch, 32, seq, 128]
k = k.repeat_interleave(4, dim=1)
v = v.repeat_interleave(4, dim=1)
```

## RoPE (Rotary Position Embeddings)

Applied **after projection**, **before attention**:

```python
# 1. Project
q = q_proj(hidden)  # [batch, seq, 4096]
k = k_proj(hidden)  # [batch, seq, 1024]

# 2. Reshape for heads
q = q.view(batch, seq, 32, 128).transpose(1, 2)
k = k.view(batch, seq, 8, 128).transpose(1, 2)

# 3. Apply RoPE
cos, sin = rotary_emb(seq_len)
q = q * cos + rotate_half(q) * sin
k = k * cos + rotate_half(k) * sin

# 4. Expand KV for GQA
k = k.repeat_interleave(4, dim=1)

# 5. Attention
attn = softmax(q @ k.T / sqrt(128))
```

## Attention Implementations

LLaMA supports multiple attention backends:

| Implementation | Class                | Notes                                                 |
| -------------- | -------------------- | ----------------------------------------------------- |
| Eager          | LlamaAttention       | Standard PyTorch, returns patterns                    |
| SDPA           | LlamaSdpaAttention   | Uses torch.nn.functional.scaled_dot_product_attention |
| Flash          | LlamaFlashAttention2 | FlashAttention-2, fastest but no patterns             |

**To get attention patterns**, you may need to set:

```python
model.config._attn_implementation = "eager"
```

## Common Issues

### Issue: Patterns are None

**Cause**: FlashAttention doesn't return attention patterns.
**Fix**: Use eager attention or capture pre-softmax logits.

### Issue: K/V shape mismatch

**Cause**: GQA - K/V have fewer heads than Q.
**Fix**: Use `expand_kv_for_gqa()` or compare with correct head counts.

### Issue: Position-dependent artifacts

**Cause**: RoPE is applied after our hooks capture Q/K.
**Fix**: Account for RoPE in analysis, or hook after RoPE application.

## LLaMA 2 vs LLaMA 3

| Property | LLaMA-2-7B | LLaMA-3-8B |
| -------- | ---------- | ---------- |
| Layers   | 32         | 32         |
| Q Heads  | 32         | 32         |
| KV Heads | 32 (MHA)   | 8 (GQA)    |
| Head Dim | 128        | 128        |
| Vocab    | 32000      | 128256     |
| Context  | 4096       | 8192       |
