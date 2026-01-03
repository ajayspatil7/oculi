"""
Head Composition Analysis
=========================

Analyze how attention heads compose and interact across layers.

This module provides methods to understand:
- Q-K composition: How one head's output affects another's attention pattern
- O-V composition: How values written by one head are read by another
- Virtual attention: Effective attention after multi-layer composition
- Path importance: Which head paths matter most for predictions

Reference:
- "A Mathematical Framework for Transformer Circuits" (Elhage et al., 2021)
- "In-context Learning and Induction Heads" (Olsson et al., 2022)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn.functional as F

from oculi.capture.structures import AttentionCapture, ResidualCapture


@dataclass
class CompositionResult:
    """
    Container for composition analysis results.
    
    Attributes:
        values: Composition scores tensor
        method: Name of composition method used
        source: Source heads (layer, head) or description
        target: Target heads (layer, head) or description
        metadata: Additional information
    """
    values: torch.Tensor
    method: str
    source: Optional[str] = None
    target: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class CompositionAnalysis:
    """
    Analyze attention head composition across layers.
    
    Transformers compute through attention head composition:
    - Layer L head can read output from any layer < L head
    - "QK-composition": Earlier head affects which tokens are attended
    - "OV-composition": Earlier head affects what information is read
    
    Example:
        >>> comp = CompositionAnalysis.qk_composition(capture, (0, 0), (2, 1))
        >>> print(f"Head 0.0 → Head 2.1 QK composition: {comp.values.mean():.4f}")
    """
    
    # =========================================================================
    # QK COMPOSITION
    # =========================================================================
    
    @staticmethod
    def qk_composition(
        capture: AttentionCapture,
        source: Tuple[int, int],
        target: Tuple[int, int],
        normalize: bool = True
    ) -> CompositionResult:
        """
        Measure Q-K composition between two heads.
        
        QK-composition occurs when a later head's queries attend to positions
        that were written to by an earlier head. This measures how much
        the source head's output affects the target head's attention pattern.
        
        Method:
        - Get attention pattern from source head: A_src [T, T]
        - For each query position in target, compute:
          QK_score = sum over k: A_src[q, k] * A_tgt[q, k]
        - This is essentially: how similar are the attention patterns?
        
        Args:
            capture: AttentionCapture with patterns
            source: (layer, head) of source head
            target: (layer, head) of target head
            normalize: Whether to normalize scores
            
        Returns:
            CompositionResult with values shape [T] (per-position scores)
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        
        src_layer, src_head = source
        tgt_layer, tgt_head = target
        
        if src_layer >= tgt_layer:
            raise ValueError(
                f"Source layer {src_layer} must be before target layer {tgt_layer}"
            )
        
        patterns = capture.patterns.float()  # [L, H, T, T]
        
        src_attn = patterns[src_layer, src_head]  # [T, T]
        tgt_attn = patterns[tgt_layer, tgt_head]  # [T, T]
        
        # Compute per-position composition score
        # For each query position, how aligned are the key distributions?
        # score[q] = sum_k(A_src[q,k] * A_tgt[q,k])
        composition = (src_attn * tgt_attn).sum(dim=-1)  # [T]
        
        if normalize:
            # Normalize to [0, 1] range
            # Max possible value is 1 (when patterns are identical)
            composition = composition.clamp(0, 1)
        
        return CompositionResult(
            values=composition.cpu(),
            method="qk_composition",
            source=f"L{src_layer}H{src_head}",
            target=f"L{tgt_layer}H{tgt_head}",
            metadata={
                "source_layer": src_layer,
                "source_head": src_head,
                "target_layer": tgt_layer,
                "target_head": tgt_head,
                "n_tokens": composition.shape[0],
            }
        )
    
    # =========================================================================
    # OV COMPOSITION
    # =========================================================================
    
    @staticmethod
    def ov_composition(
        capture: AttentionCapture,
        source: Tuple[int, int],
        target: Tuple[int, int]
    ) -> CompositionResult:
        """
        Measure O-V composition between two heads.
        
        OV-composition measures how much the output of the source head
        (values it writes) is read by the target head (affects its queries
        or keys).
        
        Method:
        - Source head writes: output = A_src @ V_src
        - Target head reads this through Q or K projections
        - Composition score = correlation between source output and target input
        
        Args:
            capture: AttentionCapture with patterns and values
            source: (layer, head) of source head
            target: (layer, head) of target head
            
        Returns:
            CompositionResult with values shape [T] 
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        if capture.values is None:
            raise ValueError("AttentionCapture must have values")
        
        src_layer, src_head = source
        tgt_layer, tgt_head = target
        
        if src_layer >= tgt_layer:
            raise ValueError(
                f"Source layer {src_layer} must be before target layer {tgt_layer}"
            )
        
        patterns = capture.patterns.float()  # [L, H, T, T]
        values = capture.values.float()  # [L, H_kv, T, D]
        
        n_kv_heads = capture.n_kv_heads
        n_heads = capture.n_heads
        
        # Handle GQA
        if n_kv_heads != n_heads:
            gqa_ratio = n_heads // n_kv_heads
            src_kv_head = src_head // gqa_ratio
        else:
            src_kv_head = src_head
        
        # Source head output: A @ V
        src_attn = patterns[src_layer, src_head]  # [T, T]
        src_values = values[src_layer, src_kv_head]  # [T, D]
        src_output = torch.mm(src_attn, src_values)  # [T, D]
        
        # Target head attention
        tgt_attn = patterns[tgt_layer, tgt_head]  # [T, T]
        
        # Composition: how much does target attend to positions
        # where source wrote important info?
        # Weight by source output magnitude
        src_magnitude = src_output.norm(dim=-1)  # [T]
        
        # For each query position: weighted attention to source outputs
        composition = torch.mv(tgt_attn, src_magnitude)  # [T]
        
        # Normalize
        composition = composition / (composition.max() + 1e-10)
        
        return CompositionResult(
            values=composition.cpu(),
            method="ov_composition",
            source=f"L{src_layer}H{src_head}",
            target=f"L{tgt_layer}H{tgt_head}",
            metadata={
                "source_layer": src_layer,
                "source_head": src_head,
                "target_layer": tgt_layer,
                "target_head": tgt_head,
            }
        )
    
    # =========================================================================
    # VIRTUAL ATTENTION
    # =========================================================================
    
    @staticmethod
    def virtual_attention(
        capture: AttentionCapture,
        heads: List[Tuple[int, int]]
    ) -> CompositionResult:
        """
        Compute effective attention through a path of heads.
        
        When heads compose, the effective attention pattern is the
        matrix product of individual attention patterns.
        
        For path [h0, h1, h2]:
        virtual_attn = A_h2 @ A_h1 @ A_h0
        
        This shows where info starting at position j ends up at position i
        after passing through all heads in the path.
        
        Args:
            capture: AttentionCapture with patterns
            heads: List of (layer, head) tuples in causal order
            
        Returns:
            CompositionResult with values shape [T, T]
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        
        if len(heads) < 2:
            raise ValueError("Need at least 2 heads for composition")
        
        # Verify causal ordering
        for i in range(len(heads) - 1):
            if heads[i][0] >= heads[i + 1][0]:
                raise ValueError(
                    f"Heads must be in causal order: {heads[i]} before {heads[i+1]}"
                )
        
        patterns = capture.patterns.float()  # [L, H, T, T]
        n_tokens = patterns.shape[2]
        
        # Start with first head's attention
        layer, head = heads[0]
        virtual = patterns[layer, head].clone()  # [T, T]
        
        # Compose through subsequent heads
        for layer, head in heads[1:]:
            attn = patterns[layer, head]  # [T, T]
            virtual = torch.mm(attn, virtual)  # [T, T]
            
            # Normalize to prevent explosion
            row_sums = virtual.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            virtual = virtual / row_sums
        
        path_str = " → ".join(f"L{l}H{h}" for l, h in heads)
        
        return CompositionResult(
            values=virtual.cpu(),
            method="virtual_attention",
            source=f"L{heads[0][0]}H{heads[0][1]}",
            target=f"L{heads[-1][0]}H{heads[-1][1]}",
            metadata={
                "path": heads,
                "path_str": path_str,
                "path_length": len(heads),
                "n_tokens": n_tokens,
            }
        )
    
    # =========================================================================
    # PATH PATCHING SCORE
    # =========================================================================
    
    @staticmethod
    def path_patching_score(
        capture: AttentionCapture,
        residual: ResidualCapture,
        path: List[Tuple[int, int]]
    ) -> CompositionResult:
        """
        Estimate importance of a path through attention heads.
        
        Approximates what activation patching would measure by computing
        the magnitude of information flow through the path.
        
        Method:
        - Compute virtual attention for path
        - Weight by residual stream magnitude changes
        - Higher score = path more important for model output
        
        Args:
            capture: AttentionCapture with patterns
            residual: ResidualCapture for magnitude weighting
            path: List of (layer, head) tuples
            
        Returns:
            CompositionResult with scalar importance score
        """
        if len(path) < 2:
            raise ValueError("Need at least 2 heads for path")
        
        # Get virtual attention for path
        virtual = CompositionAnalysis.virtual_attention(capture, path)
        virtual_attn = virtual.values  # [T, T]
        
        # Weight by residual magnitude at start and end layers
        start_layer = path[0][0]
        end_layer = path[-1][0]
        
        if residual.post_mlp is not None:
            start_resid = residual.post_mlp[start_layer].float()  # [T, H]
            end_resid = residual.post_mlp[end_layer].float()  # [T, H]
            
            start_magnitude = start_resid.norm(dim=-1)  # [T]
            end_magnitude = end_resid.norm(dim=-1)  # [T]
            
            # Score: how much info from high-magnitude positions
            # flows to high-magnitude outputs?
            weighted = virtual_attn * start_magnitude.unsqueeze(0)  # [T, T]
            score = (weighted.sum(dim=-1) * end_magnitude).mean()
        else:
            # Fallback: just use virtual attention magnitude
            score = virtual_attn.abs().mean()
        
        path_str = " → ".join(f"L{l}H{h}" for l, h in path)
        
        return CompositionResult(
            values=torch.tensor([score]).cpu(),
            method="path_patching_score",
            source=path_str,
            target="output",
            metadata={
                "path": path,
                "path_str": path_str,
            }
        )
    
    # =========================================================================
    # COMPOSITION MATRIX
    # =========================================================================
    
    @staticmethod
    def composition_matrix(
        capture: AttentionCapture,
        method: str = "qk"
    ) -> CompositionResult:
        """
        Compute full head-to-head composition matrix.
        
        Creates a (L×H) x (L×H) matrix where entry [i,j] measures
        how much head j (earlier) composes with head i (later).
        
        Args:
            capture: AttentionCapture with patterns (and values for ov)
            method: "qk" for QK-composition, "ov" for OV-composition
            
        Returns:
            CompositionResult with values shape [L*H, L*H]
            values[i, j] = composition from head j to head i
            (j must be in earlier layer than i for non-zero)
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        
        n_layers = capture.n_layers
        n_heads = capture.n_heads
        total_heads = n_layers * n_heads
        
        matrix = torch.zeros(total_heads, total_heads)
        
        for tgt_layer in range(n_layers):
            for tgt_head in range(n_heads):
                tgt_idx = tgt_layer * n_heads + tgt_head
                
                for src_layer in range(tgt_layer):  # Only earlier layers
                    for src_head in range(n_heads):
                        src_idx = src_layer * n_heads + src_head
                        
                        try:
                            if method == "qk":
                                result = CompositionAnalysis.qk_composition(
                                    capture, (src_layer, src_head), (tgt_layer, tgt_head)
                                )
                            elif method == "ov":
                                result = CompositionAnalysis.ov_composition(
                                    capture, (src_layer, src_head), (tgt_layer, tgt_head)
                                )
                            else:
                                raise ValueError(f"Unknown method: {method}")
                            
                            # Use mean composition score
                            matrix[tgt_idx, src_idx] = result.values.mean()
                        except Exception:
                            # Skip invalid pairs
                            pass
        
        return CompositionResult(
            values=matrix,
            method=f"composition_matrix_{method}",
            source="all_heads",
            target="all_heads",
            metadata={
                "n_layers": n_layers,
                "n_heads": n_heads,
                "composition_method": method,
            }
        )
    
    # =========================================================================
    # INDUCTION HEAD COMPOSITION
    # =========================================================================
    
    @staticmethod
    def detect_induction_circuit(
        capture: AttentionCapture,
        threshold: float = 0.3
    ) -> CompositionResult:
        """
        Detect potential induction head circuits.
        
        Induction heads typically work in pairs:
        - "Previous token head": attends to t-1 positions
        - "Induction head": composes with previous token head
        
        This finds head pairs that show strong composition patterns
        consistent with induction behavior.
        
        Args:
            capture: AttentionCapture with patterns
            threshold: Minimum composition score for detection
            
        Returns:
            CompositionResult with detected circuits
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        
        patterns = capture.patterns.float()  # [L, H, T, T]
        n_layers = capture.n_layers
        n_heads = capture.n_heads
        n_tokens = capture.n_tokens
        
        circuits = []
        
        # Look for pairs where:
        # 1. Early head has strong diagonal-offset (previous token) pattern
        # 2. Later head strongly composes with it
        
        for early_layer in range(n_layers - 1):
            for early_head in range(n_heads):
                early_attn = patterns[early_layer, early_head]  # [T, T]
                
                # Check if this is a previous-token head
                prev_token_score = 0.0
                for t in range(1, n_tokens):
                    prev_token_score += early_attn[t, t-1].item()
                prev_token_score /= (n_tokens - 1)
                
                if prev_token_score < threshold:
                    continue  # Not a previous-token head
                
                # Look for later heads that compose with this
                for late_layer in range(early_layer + 1, n_layers):
                    for late_head in range(n_heads):
                        comp = CompositionAnalysis.qk_composition(
                            capture, (early_layer, early_head), (late_layer, late_head)
                        )
                        comp_score = comp.values.mean().item()
                        
                        if comp_score > threshold:
                            circuits.append({
                                "previous_token_head": (early_layer, early_head),
                                "induction_head": (late_layer, late_head),
                                "prev_token_score": prev_token_score,
                                "composition_score": comp_score,
                            })
        
        # Sort by composition score
        circuits.sort(key=lambda x: -x["composition_score"])
        
        return CompositionResult(
            values=torch.tensor([len(circuits)]),
            method="detect_induction_circuit",
            source="previous_token_heads",
            target="induction_heads",
            metadata={
                "circuits": circuits,
                "threshold": threshold,
            }
        )
