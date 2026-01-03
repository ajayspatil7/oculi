"""
Attribution Methods
===================

Causal attribution methods for understanding information flow in transformers.

This module provides methods to:
- Track attention flow across layers
- Weight attention by value contribution  
- Attribute logits to layers and components
- Decompose contributions by attention vs MLP

Reference:
- "Attention is Not All You Need" (Kobayashi et al., 2020)
- "Transformer Circuits" (Elhage et al., 2021)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn.functional as F

from oculi.capture.structures import (
    AttentionCapture,
    ResidualCapture,
    MLPCapture,
    LogitCapture,
)


@dataclass
class AttributionResult:
    """
    Container for attribution results.
    
    Attributes:
        values: Attribution scores tensor
        method: Name of attribution method used
        target: What was attributed (e.g., token, layer, position)
        metadata: Additional information about the attribution
    """
    values: torch.Tensor
    method: str
    target: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class AttributionMethods:
    """
    Causal attribution methods for transformer internals.
    
    All methods are static and operate on capture objects.
    
    Example:
        >>> full = adapter.capture_full(input_ids, residual_config=ResidualConfig())
        >>> unembed = model.lm_head.weight
        >>> dla = AttributionMethods.direct_logit_attribution(
        ...     full.residual, unembed, target_token_id=42
        ... )
        >>> print(f"Top contributor: Layer {dla.values.argmax()}")
    """
    
    # =========================================================================
    # ATTENTION FLOW
    # =========================================================================
    
    @staticmethod
    def attention_flow(
        capture: AttentionCapture,
        residual: Optional[ResidualCapture] = None,
        normalize: bool = True
    ) -> AttributionResult:
        """
        Track information flow through attention across layers.
        
        Computes how information from source positions propagates to target
        positions through the attention mechanism across layers.
        
        Method:
        - Layer 0: flow = attention_pattern (direct)
        - Layer L: flow[L] = attention[L] @ flow[L-1] (composition)
        - Optionally weight by residual stream magnitude changes
        
        Args:
            capture: AttentionCapture with patterns [L, H, T, T]
            residual: ResidualCapture for weighting (optional)
            normalize: Whether to normalize flow matrices per layer
            
        Returns:
            AttributionResult with values shape [L, H, T, T] representing
            cumulative attention flow from source (dim -1) to target (dim -2)
            
        Example:
            >>> flow = AttributionMethods.attention_flow(capture)
            >>> # flow.values[L, H, i, j] = how much position j contributes to position i at layer L
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        
        patterns = capture.patterns  # [L, H, T, T]
        n_layers, n_heads, n_tokens, _ = patterns.shape
        
        # Initialize cumulative flow
        # Use float32 for numerical stability
        flow = torch.zeros_like(patterns, dtype=torch.float32)
        
        # Layer 0: direct attention
        flow[0] = patterns[0].float()
        
        # Subsequent layers: compose attention through previous layers
        for l in range(1, n_layers):
            # flow[l] = attn[l] @ flow[l-1]
            # This computes: "where does layer l attend, weighted by cumulative history"
            attn_l = patterns[l].float()  # [H, T, T]
            flow_prev = flow[l - 1]  # [H, T, T]
            
            # Matrix multiply per head
            # attn_l[h, i, j] @ flow_prev[h, j, k] -> flow[h, i, k]
            flow[l] = torch.bmm(
                attn_l.view(-1, n_tokens, n_tokens),
                flow_prev.view(-1, n_tokens, n_tokens)
            ).view(n_heads, n_tokens, n_tokens)
            
            if normalize:
                # Normalize rows to sum to 1 (prevent values exploding)
                row_sums = flow[l].sum(dim=-1, keepdim=True).clamp(min=1e-10)
                flow[l] = flow[l] / row_sums
        
        # Optionally weight by residual stream changes
        if residual is not None and residual.post_mlp is not None:
            # Weight by magnitude change in residual stream
            residual_norms = residual.post_mlp.norm(dim=-1)  # [L, T]
            # Broadcast to [L, 1, T, 1] for weighting
            weights = residual_norms.unsqueeze(1).unsqueeze(-1)
            weights = weights / weights.mean()  # Normalize
            flow = flow * weights.to(flow.dtype)
        
        return AttributionResult(
            values=flow.cpu(),
            method="attention_flow",
            target=None,
            metadata={
                "n_layers": n_layers,
                "n_heads": n_heads,
                "n_tokens": n_tokens,
                "normalized": normalize,
            }
        )
    
    # =========================================================================
    # VALUE-WEIGHTED ATTENTION
    # =========================================================================
    
    @staticmethod
    def value_weighted_attention(
        capture: AttentionCapture,
        norm_type: str = "l2"
    ) -> AttributionResult:
        """
        Compute attention weighted by value vector contribution.
        
        Raw attention patterns don't account for value magnitude.
        A head might attend strongly but write small values.
        This weights attention by how much each attended position
        actually contributes to the output.
        
        Method:
        - Compute value norms: ||V[t]|| for each token t
        - Weight attention: weighted[t_q, t_k] = attn[t_q, t_k] * ||V[t_k]||
        - Normalize so weights sum to 1
        
        Args:
            capture: AttentionCapture with patterns and values
            norm_type: How to compute value magnitude ("l2", "l1", "linf")
            
        Returns:
            AttributionResult with values shape [L, H, T, T]
            Attention patterns reweighted by value contribution
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        if capture.values is None:
            raise ValueError("AttentionCapture must have values")
        
        patterns = capture.patterns.float()  # [L, H, T, T]
        values = capture.values.float()  # [L, H_kv, T, D]
        
        n_layers, n_heads, n_tokens, _ = patterns.shape
        n_kv_heads = capture.n_kv_heads
        
        # Compute value norms per token
        if norm_type == "l2":
            value_norms = values.norm(dim=-1)  # [L, H_kv, T]
        elif norm_type == "l1":
            value_norms = values.abs().sum(dim=-1)
        elif norm_type == "linf":
            value_norms = values.abs().max(dim=-1).values
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        
        # Handle GQA: expand KV heads to match query heads
        if n_kv_heads != n_heads:
            gqa_ratio = n_heads // n_kv_heads
            value_norms = value_norms.repeat_interleave(gqa_ratio, dim=1)
        
        # Weight attention by value norms
        # value_norms: [L, H, T] -> [L, H, 1, T] for broadcasting
        weights = value_norms.unsqueeze(2)  # [L, H, 1, T]
        weighted = patterns * weights  # [L, H, T, T]
        
        # Normalize rows to sum to 1
        row_sums = weighted.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        weighted = weighted / row_sums
        
        return AttributionResult(
            values=weighted.cpu(),
            method="value_weighted_attention",
            target=None,
            metadata={
                "norm_type": norm_type,
                "n_layers": n_layers,
                "n_heads": n_heads,
            }
        )
    
    # =========================================================================
    # DIRECT LOGIT ATTRIBUTION
    # =========================================================================
    
    @staticmethod
    def direct_logit_attribution(
        residual: ResidualCapture,
        unembed_matrix: torch.Tensor,
        target_token_id: int,
        position: int = -1
    ) -> AttributionResult:
        """
        Compute direct effect of each layer's residual on target logit.
        
        Measures how much each layer's contribution to residual stream
        directly affects the logit of a specific token.
        
        Method:
        - Compute layer contribution: delta[L] = post_mlp[L] - post_mlp[L-1]
        - Project onto target token: attribution[L] = delta[L] @ unembed[target]
        
        Args:
            residual: ResidualCapture with post_mlp activations
            unembed_matrix: Model's unembedding matrix (lm_head.weight) [V, H] or [H, V]
            target_token_id: Vocabulary index of target token
            position: Sequence position to analyze (-1 = last)
            
        Returns:
            AttributionResult with values shape [L]
            Each value = layer's direct contribution to target logit
            
        Example:
            >>> dla = AttributionMethods.direct_logit_attribution(
            ...     residual, unembed, target_token_id=tokenizer.encode("cat")[0]
            ... )
            >>> print(f"Most important layer: {dla.values.argmax().item()}")
        """
        if residual.post_mlp is None:
            raise ValueError("ResidualCapture must have post_mlp")
        
        post_mlp = residual.post_mlp.float()  # [L, T, H]
        n_layers, n_tokens, hidden_dim = post_mlp.shape
        
        # Handle negative indexing
        if position < 0:
            position = n_tokens + position
        
        # Get unembedding vector for target token
        unembed_vec = AttributionMethods._get_unembed_vector(
            unembed_matrix, target_token_id, hidden_dim
        )
        
        # Extract residuals at the target position
        residuals_at_pos = post_mlp[:, position, :]  # [L, H]
        
        # Compute layer deltas
        # delta[0] = post_mlp[0] (first layer contribution)
        # delta[L] = post_mlp[L] - post_mlp[L-1] (subsequent layers)
        deltas = torch.zeros(n_layers, hidden_dim, dtype=post_mlp.dtype, device=post_mlp.device)
        deltas[0] = residuals_at_pos[0]
        for l in range(1, n_layers):
            deltas[l] = residuals_at_pos[l] - residuals_at_pos[l - 1]
        
        # Project onto target token's unembedding direction
        attributions = torch.mv(deltas, unembed_vec.to(deltas.device))  # [L]
        
        return AttributionResult(
            values=attributions.cpu(),
            method="direct_logit_attribution",
            target=f"token_{target_token_id}",
            metadata={
                "target_token_id": target_token_id,
                "position": position,
                "n_layers": n_layers,
            }
        )
    
    # =========================================================================
    # COMPONENT ATTRIBUTION
    # =========================================================================
    
    @staticmethod
    def component_attribution(
        residual: ResidualCapture,
        mlp: Optional[MLPCapture],
        unembed_matrix: torch.Tensor,
        target_token_id: int,
        position: int = -1
    ) -> AttributionResult:
        """
        Decompose logit attribution into attention vs MLP components.
        
        Method:
        - attn_contrib[L] = (post_attn[L] - pre_attn[L]) @ unembed[target]
        - mlp_contrib[L] = (post_mlp[L] - pre_mlp[L]) @ unembed[target]
        
        Args:
            residual: ResidualCapture with all four positions
            mlp: MLPCapture (optional, for validation)
            unembed_matrix: Model's unembedding matrix
            target_token_id: Target token vocabulary index
            position: Sequence position to analyze
            
        Returns:
            AttributionResult with values shape [L, 2]
            values[:, 0] = attention contribution
            values[:, 1] = MLP contribution
            metadata contains {'components': ['attention', 'mlp']}
        """
        if residual.pre_attn is None or residual.post_attn is None:
            raise ValueError("ResidualCapture must have pre_attn and post_attn")
        if residual.pre_mlp is None or residual.post_mlp is None:
            raise ValueError("ResidualCapture must have pre_mlp and post_mlp")
        
        pre_attn = residual.pre_attn.float()   # [L, T, H]
        post_attn = residual.post_attn.float()  # [L, T, H]
        pre_mlp = residual.pre_mlp.float()     # [L, T, H]
        post_mlp = residual.post_mlp.float()   # [L, T, H]
        
        n_layers, n_tokens, hidden_dim = pre_attn.shape
        
        # Handle negative indexing
        if position < 0:
            position = n_tokens + position
        
        # Get unembedding vector
        unembed_vec = AttributionMethods._get_unembed_vector(
            unembed_matrix, target_token_id, hidden_dim
        ).to(pre_attn.device)
        
        # Compute attention contribution per layer
        attn_delta = post_attn[:, position, :] - pre_attn[:, position, :]  # [L, H]
        attn_contrib = torch.mv(attn_delta, unembed_vec)  # [L]
        
        # Compute MLP contribution per layer
        mlp_delta = post_mlp[:, position, :] - pre_mlp[:, position, :]  # [L, H]
        mlp_contrib = torch.mv(mlp_delta, unembed_vec)  # [L]
        
        # Stack into [L, 2]
        attributions = torch.stack([attn_contrib, mlp_contrib], dim=1)
        
        return AttributionResult(
            values=attributions.cpu(),
            method="component_attribution",
            target=f"token_{target_token_id}",
            metadata={
                "components": ["attention", "mlp"],
                "target_token_id": target_token_id,
                "position": position,
                "n_layers": n_layers,
            }
        )
    
    # =========================================================================
    # HEAD ATTRIBUTION
    # =========================================================================
    
    @staticmethod
    def head_attribution(
        capture: AttentionCapture,
        output_weights: torch.Tensor,
        unembed_matrix: torch.Tensor,
        target_token_id: int,
        position: int = -1
    ) -> AttributionResult:
        """
        Attribute target logit to individual attention heads.
        
        Method:
        - For each head, compute: head_output = attn_pattern @ V @ W_O
        - Project onto target: attribution[L, H] = head_output @ unembed[target]
        
        This is more granular than component_attribution.
        
        Args:
            capture: AttentionCapture with patterns and values
            output_weights: W_O matrices [L, H, head_dim, hidden_dim] or similar
            unembed_matrix: Unembedding matrix
            target_token_id: Target token
            position: Sequence position
            
        Returns:
            AttributionResult with values shape [L, H]
            Each value = head's direct contribution to target logit
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        if capture.values is None:
            raise ValueError("AttentionCapture must have values")
        
        patterns = capture.patterns.float()  # [L, H, T, T]
        values = capture.values.float()  # [L, H_kv, T, D]
        
        n_layers, n_heads, n_tokens, _ = patterns.shape
        n_kv_heads = capture.n_kv_heads
        head_dim = capture.head_dim
        hidden_dim = n_heads * head_dim
        
        # Handle negative indexing
        if position < 0:
            position = n_tokens + position
        
        # Get unembedding vector
        unembed_vec = AttributionMethods._get_unembed_vector(
            unembed_matrix, target_token_id, hidden_dim
        )
        
        # Handle GQA: expand V to match Q heads
        if n_kv_heads != n_heads:
            gqa_ratio = n_heads // n_kv_heads
            values = values.repeat_interleave(gqa_ratio, dim=1)  # [L, H, T, D]
        
        attributions = torch.zeros(n_layers, n_heads)
        
        for l in range(n_layers):
            for h in range(n_heads):
                # Get attention weights for this head at target position
                attn = patterns[l, h, position, :]  # [T]
                
                # Weighted sum of values
                v = values[l, h]  # [T, D]
                head_output = torch.mv(v.T, attn)  # [D]
                
                # Apply output projection if provided
                if output_weights.dim() == 4:
                    # [L, H, D, hidden_dim]
                    w_o = output_weights[l, h]  # [D, hidden_dim]
                    projected = torch.mv(w_o.T, head_output)  # [hidden_dim]
                else:
                    # Fallback: assume head_dim portion of hidden
                    projected = torch.zeros(hidden_dim)
                    start = h * head_dim
                    projected[start:start + head_dim] = head_output
                
                # Project onto target
                attributions[l, h] = torch.dot(
                    projected.to(unembed_vec.device), 
                    unembed_vec
                )
        
        return AttributionResult(
            values=attributions.cpu(),
            method="head_attribution",
            target=f"token_{target_token_id}",
            metadata={
                "target_token_id": target_token_id,
                "position": position,
                "n_layers": n_layers,
                "n_heads": n_heads,
            }
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    @staticmethod
    def _get_unembed_vector(
        unembed_matrix: torch.Tensor,
        token_id: int,
        hidden_dim: int
    ) -> torch.Tensor:
        """
        Extract unembedding vector for a token.
        
        Handles both [V, H] and [H, V] shaped matrices.
        """
        if unembed_matrix.shape[0] == hidden_dim:
            # Shape is [H, V]
            return unembed_matrix[:, token_id]
        elif unembed_matrix.shape[1] == hidden_dim:
            # Shape is [V, H]
            return unembed_matrix[token_id, :]
        else:
            raise ValueError(
                f"Cannot determine unembed_matrix orientation. "
                f"Shape {unembed_matrix.shape}, hidden_dim={hidden_dim}"
            )
    
    @staticmethod
    def _compute_layer_deltas(
        residual: ResidualCapture,
        position: int
    ) -> torch.Tensor:
        """
        Compute per-layer residual changes.
        
        Args:
            residual: ResidualCapture with post_mlp
            position: Sequence position
            
        Returns:
            [L, H] tensor of per-layer contributions
        """
        if residual.post_mlp is None:
            raise ValueError("ResidualCapture must have post_mlp")
        
        post_mlp = residual.post_mlp.float()  # [L, T, H]
        n_layers, n_tokens, hidden_dim = post_mlp.shape
        
        if position < 0:
            position = n_tokens + position
        
        residuals = post_mlp[:, position, :]  # [L, H]
        
        deltas = torch.zeros_like(residuals)
        deltas[0] = residuals[0]
        for l in range(1, n_layers):
            deltas[l] = residuals[l] - residuals[l - 1]
        
        return deltas
    
    @staticmethod
    def top_attributions(
        result: AttributionResult,
        k: int = 10
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """
        Get top-k attributions with their indices.
        
        Args:
            result: AttributionResult from any method
            k: Number of top attributions to return
            
        Returns:
            [(index_tuple, value), ...] sorted by |value| descending
        """
        values = result.values.flatten()
        abs_values = values.abs()
        
        k = min(k, len(values))
        top_indices = torch.topk(abs_values, k).indices
        
        results = []
        shape = result.values.shape
        for idx in top_indices:
            # Convert flat index to multi-dimensional index
            multi_idx = []
            remaining = idx.item()
            for dim in reversed(shape):
                multi_idx.append(remaining % dim)
                remaining //= dim
            multi_idx = tuple(reversed(multi_idx))
            
            results.append((multi_idx, values[idx].item()))
        
        return results
