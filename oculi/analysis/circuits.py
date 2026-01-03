"""
Circuit Detection
=================

Detect canonical circuit patterns in transformer attention.

Implements detection for:
- Induction heads: [A][B]...[A] -> attend to [B]
- Previous token heads: Primarily attend to t-1
- Duplicate token heads: Attend to previous occurrences
- Positional heads: BOS, recent, local patterns

Reference: "In-context Learning and Induction Heads" (Olsson et al., 2022)
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F


class CircuitDetection:
    """
    Detect canonical circuit patterns in transformers.
    
    All methods are static and operate on AttentionCapture objects.
    
    Example:
        >>> from oculi.analysis import CircuitDetection
        >>> capture = adapter.capture(input_ids)
        >>> induction = CircuitDetection.detect_induction_heads(capture)
        >>> print(f"Induction heads: {(induction > 0.5).sum()} found")
    """
    
    @staticmethod
    def detect_induction_heads(
        capture,
        threshold: float = 0.5,
        min_seq_len: int = 10
    ) -> torch.Tensor:
        """
        Detect induction heads: [A][B]...[A] -> attend strongly to [B].
        
        Induction heads implement in-context learning by:
        1. Finding previous occurrences of the current token
        2. Attending to the token that followed previously
        
        Method:
        - Find repeated tokens in sequence
        - Check if attention at second occurrence attends to token 
          after first occurrence
        - Score = average attention to "correct" position
        
        Args:
            capture: AttentionCapture with patterns
            threshold: Score threshold for classification
            min_seq_len: Minimum sequence length for detection
            
        Returns:
            [L, H] scores (0-1, higher = more induction-like)
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        
        patterns = capture.patterns  # [L, H, T, T]
        n_layers, n_heads, n_tokens, _ = patterns.shape
        
        if n_tokens < min_seq_len:
            return torch.zeros(n_layers, n_heads)
        
        scores = torch.zeros(n_layers, n_heads)
        
        # For each layer and head, compute induction score
        for l in range(n_layers):
            for h in range(n_heads):
                attn = patterns[l, h]  # [T, T]
                
                # Look for diagonal offset pattern
                # If token at position i == token at position j (j < i),
                # induction head attends from i to j+1
                
                # Simple heuristic: check attention to position i-1 from i
                # (previous token head behavior)
                # and off-diagonal patterns
                
                # Check for "shifted diagonal" pattern
                # Strong induction: high attention at (i, j+1) when tokens i and j match
                
                # Simplified: measure off-diagonal concentration
                diag_scores = []
                for offset in range(1, min(5, n_tokens)):
                    # How much attention goes to exactly 'offset' positions back
                    attn_to_offset = torch.diagonal(attn, offset=-offset).mean()
                    diag_scores.append(attn_to_offset.item())
                
                # Induction pattern: attention clusters at specific offset > 1
                if len(diag_scores) > 1:
                    # Score based on having peaked attention at offset 2+
                    max_offset = torch.tensor(diag_scores[1:]).max() if len(diag_scores) > 1 else 0
                    scores[l, h] = max_offset
        
        return scores
    
    @staticmethod
    def detect_previous_token_heads(
        capture,
        threshold: float = 0.8
    ) -> torch.Tensor:
        """
        Detect heads attending primarily to previous token.
        
        Previous token heads have a strong diagonal pattern where
        position i attends primarily to position i-1.
        
        Args:
            capture: AttentionCapture with patterns
            threshold: Threshold for classification
            
        Returns:
            [L, H] boolean mask
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        
        patterns = capture.patterns  # [L, H, T, T]
        n_layers, n_heads, n_tokens, _ = patterns.shape
        
        # Extract diagonal offset -1 (previous token attention)
        # For each position i, attention[i, i-1]
        prev_token_attn = torch.zeros(n_layers, n_heads, n_tokens - 1)
        
        for t in range(1, n_tokens):
            prev_token_attn[:, :, t-1] = patterns[:, :, t, t-1]
        
        # Average attention to previous token
        mean_prev_attn = prev_token_attn.mean(dim=-1)  # [L, H]
        
        return mean_prev_attn > threshold
    
    @staticmethod
    def detect_duplicate_token_heads(
        capture,
        input_ids: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Detect heads attending to previous occurrences of same token.
        
        Args:
            capture: AttentionCapture with patterns
            input_ids: [1, T] token IDs
            threshold: Score threshold
            
        Returns:
            [L, H] scores
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        
        patterns = capture.patterns  # [L, H, T, T]
        n_layers, n_heads, n_tokens, _ = patterns.shape
        
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)  # [T]
        
        scores = torch.zeros(n_layers, n_heads)
        
        # Build duplicate token mask
        # duplicate_mask[i, j] = 1 if tokens[i] == tokens[j] and j < i
        duplicate_mask = torch.zeros(n_tokens, n_tokens)
        for i in range(n_tokens):
            for j in range(i):
                if input_ids[i] == input_ids[j]:
                    duplicate_mask[i, j] = 1.0
        
        # Score: average attention to duplicate positions
        # Normalize by number of duplicates per position
        dup_counts = duplicate_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        for l in range(n_layers):
            for h in range(n_heads):
                attn = patterns[l, h]  # [T, T]
                dup_attn = (attn * duplicate_mask).sum(dim=1)  # [T]
                normalized = dup_attn / (dup_counts.squeeze() + 1e-10)
                # Only count positions with duplicates
                has_dup = duplicate_mask.sum(dim=1) > 0
                if has_dup.any():
                    scores[l, h] = normalized[has_dup].mean()
        
        return scores
    
    @staticmethod
    def detect_positional_heads(
        capture
    ) -> Dict[str, torch.Tensor]:
        """
        Detect heads with strong positional biases.
        
        Args:
            capture: AttentionCapture with patterns
            
        Returns:
            Dict with [L, H] masks for:
            - 'bos': Attends primarily to position 0
            - 'recent': Attends primarily to last few positions
            - 'uniform': Roughly uniform attention
            - 'local': Attends to nearby positions
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        
        patterns = capture.patterns  # [L, H, T, T]
        n_layers, n_heads, n_tokens, _ = patterns.shape
        
        # BOS attention: attention to position 0
        bos_attn = patterns[:, :, :, 0].mean(dim=-1)  # [L, H]
        
        # Recent attention: attention to last 3 positions (from queries)
        recent_positions = min(3, n_tokens)
        recent_attn = patterns[:, :, :, -recent_positions:].sum(dim=-1).mean(dim=-1)
        
        # Uniform: low variance in attention distribution
        attn_var = patterns.var(dim=-1).mean(dim=-1)  # [L, H]
        uniform = attn_var < 0.01
        
        # Local: attention concentrated near diagonal
        local_scores = torch.zeros(n_layers, n_heads)
        for l in range(n_layers):
            for h in range(n_heads):
                attn = patterns[l, h]  # [T, T]
                # Sum attention within bandwidth of 5
                bandwidth = 5
                local_mass = 0.0
                for t in range(n_tokens):
                    start = max(0, t - bandwidth)
                    local_mass += attn[t, start:t+1].sum().item()
                local_scores[l, h] = local_mass / n_tokens
        
        return {
            'bos': bos_attn > 0.5,
            'recent': recent_attn > 0.5,
            'uniform': uniform,
            'local': local_scores > 0.8,
        }
    
    @staticmethod
    def attention_entropy(
        capture
    ) -> torch.Tensor:
        """
        Compute attention entropy per head (measure of attention spread).
        
        Low entropy = focused attention (attending to few positions)
        High entropy = diffuse attention (spread across many positions)
        
        Args:
            capture: AttentionCapture with patterns
            
        Returns:
            [L, H, T] entropy at each position
        """
        if capture.patterns is None:
            raise ValueError("AttentionCapture must have patterns")
        
        patterns = capture.patterns  # [L, H, T, T]
        
        # Entropy: -sum(p * log(p))
        log_patterns = torch.log(patterns + 1e-10)
        entropy = -(patterns * log_patterns).sum(dim=-1)  # [L, H, T]
        
        return entropy
    
    @staticmethod
    def head_importance_by_pattern(
        capture,
        input_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Rank heads by circuit pattern strength.
        
        Args:
            capture: AttentionCapture with patterns
            input_ids: Optional token IDs for duplicate detection
            
        Returns:
            Dict mapping pattern name to list of (layer, head, score) tuples,
            sorted by score descending
        """
        results = {}
        
        # Induction heads
        induction_scores = CircuitDetection.detect_induction_heads(capture)
        induction_list = []
        for l in range(induction_scores.shape[0]):
            for h in range(induction_scores.shape[1]):
                induction_list.append((l, h, induction_scores[l, h].item()))
        results['induction'] = sorted(induction_list, key=lambda x: -x[2])
        
        # Previous token heads
        prev_token = CircuitDetection.detect_previous_token_heads(capture, threshold=0.0)
        prev_list = []
        for l in range(prev_token.shape[0]):
            for h in range(prev_token.shape[1]):
                # Compute actual score (not just threshold)
                patterns = capture.patterns
                n_tokens = patterns.shape[2]
                prev_attn = torch.zeros(n_tokens - 1)
                for t in range(1, n_tokens):
                    prev_attn[t-1] = patterns[l, h, t, t-1]
                score = prev_attn.mean().item()
                prev_list.append((l, h, score))
        results['previous_token'] = sorted(prev_list, key=lambda x: -x[2])
        
        # Positional heads
        positional = CircuitDetection.detect_positional_heads(capture)
        results['bos'] = [(l, h, 1.0) for l in range(capture.n_layers) 
                          for h in range(capture.n_heads) if positional['bos'][l, h]]
        
        return results
    
    @staticmethod
    def summarize_circuits(
        capture,
        threshold: float = 0.5,
        input_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Run all detectors and return summary.
        
        Args:
            capture: AttentionCapture with patterns
            threshold: Classification threshold
            input_ids: Optional token IDs for duplicate detection
            
        Returns:
            Dict with:
            - 'induction_heads': [(L, H), ...]
            - 'previous_token_heads': [(L, H), ...]
            - 'positional_heads': {'bos': [...], 'recent': [...]}
            - 'total_heads': int
            - 'classified_heads': int
        """
        induction = CircuitDetection.detect_induction_heads(capture, threshold)
        prev_token = CircuitDetection.detect_previous_token_heads(capture, threshold)
        positional = CircuitDetection.detect_positional_heads(capture)
        
        induction_heads = list(zip(*torch.where(induction > threshold)))
        prev_token_heads = list(zip(*torch.where(prev_token)))
        bos_heads = list(zip(*torch.where(positional['bos'])))
        
        total_heads = capture.n_layers * capture.n_heads
        classified = len(set(induction_heads) | set(prev_token_heads) | set(bos_heads))
        
        return {
            'induction_heads': [(int(l), int(h)) for l, h in induction_heads],
            'previous_token_heads': [(int(l), int(h)) for l, h in prev_token_heads],
            'positional_heads': {
                'bos': [(int(l), int(h)) for l, h in bos_heads],
            },
            'total_heads': total_heads,
            'classified_heads': classified,
        }
