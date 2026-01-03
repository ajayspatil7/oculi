"""
Logit Lens Analysis
===================

Analyze layer-by-layer predictions using the logit lens technique.

The logit lens technique applies the unembedding matrix to intermediate
layer representations to see what the model would predict at each layer.
This reveals how the model's predictions evolve through the layers.

Reference: "Interpreting GPT: the Logit Lens" (Nostalgebraist, 2020)
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F


class LogitLensAnalysis:
    """
    Analyze layer-by-layer predictions via logit lens.
    
    The logit lens applies the language model head to intermediate
    layer representations to see what the model would predict at
    each layer.
    
    Example:
        >>> lens = LogitLensAnalysis(tokenizer)
        >>> predictions = lens.layer_predictions(logit_capture, position=-1)
        >>> for pred in predictions:
        ...     print(f"Layer {pred['layer']}: {pred['predictions'][:3]}")
    """
    
    def __init__(self, tokenizer):
        """
        Initialize with tokenizer for decoding.
        
        Args:
            tokenizer: HuggingFace tokenizer for decoding tokens
        """
        self.tokenizer = tokenizer
    
    def layer_predictions(
        self,
        logit_capture,
        token_position: int,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top-k predictions at each layer for a token position.
        
        Args:
            logit_capture: LogitCapture from adapter.capture_logits()
            token_position: Which token position to analyze (-1 for last)
            top_k: Number of top predictions to return per layer
            
        Returns:
            List of dicts, one per layer:
            [{
                'layer': int,
                'predictions': [(token_str, probability), ...],
                'top_token': str,
                'top_prob': float
            }]
        """
        results = []
        
        # Get logits - either full or from top_k capture
        if logit_capture.logits is not None:
            logits = logit_capture.logits[:, token_position, :]  # [L, V]
        elif logit_capture.top_k_logits is not None:
            # Use pre-computed top-k
            logits = logit_capture.top_k_logits[:, token_position, :]  # [L, K]
            indices = logit_capture.top_k_indices[:, token_position, :]  # [L, K]
        else:
            raise ValueError("LogitCapture has no logits data")
        
        n_layers = logits.shape[0]
        
        for layer_idx in range(n_layers):
            layer_logits = logits[layer_idx]  # [V] or [K]
            probs = F.softmax(layer_logits, dim=-1)
            
            if logit_capture.logits is not None:
                # Full logits - compute top-k
                top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[0]))
            else:
                # Already have top-k
                top_probs = probs[:top_k]
                top_indices = indices[layer_idx][:top_k]
            
            predictions = []
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                token_str = self.tokenizer.decode([idx])
                predictions.append((token_str, prob))
            
            actual_layer = logit_capture.captured_layers[layer_idx]
            results.append({
                'layer': actual_layer,
                'predictions': predictions,
                'top_token': predictions[0][0] if predictions else "",
                'top_prob': predictions[0][1] if predictions else 0.0,
            })
        
        return results
    
    def prediction_trajectory(
        self,
        logit_capture,
        token_id: int,
        position: int
    ) -> torch.Tensor:
        """
        Track specific token's probability across layers.
        
        Useful for seeing when a particular prediction emerges.
        
        Args:
            logit_capture: LogitCapture from adapter.capture_logits()
            token_id: Token ID to track
            position: Token position to analyze
            
        Returns:
            [L] tensor of probabilities at each layer
        """
        if logit_capture.logits is None:
            raise ValueError("Need full logits for trajectory tracking")
        
        logits = logit_capture.logits[:, position, :]  # [L, V]
        probs = F.softmax(logits, dim=-1)  # [L, V]
        
        return probs[:, token_id]  # [L]
    
    def convergence_layer(
        self,
        logit_capture,
        position: int,
        threshold: float = 0.9
    ) -> int:
        """
        Find layer where final prediction becomes dominant.
        
        Identifies the layer where the model first becomes "confident"
        in its final prediction.
        
        Args:
            logit_capture: LogitCapture from adapter.capture_logits()
            position: Token position to analyze
            threshold: Probability threshold for "confidence"
            
        Returns:
            Layer index where top prediction probability exceeds threshold,
            or -1 if never exceeds threshold
        """
        if logit_capture.logits is None:
            raise ValueError("Need full logits for convergence analysis")
        
        logits = logit_capture.logits[:, position, :]  # [L, V]
        probs = F.softmax(logits, dim=-1)  # [L, V]
        
        # Get the final prediction
        final_top_idx = logits[-1].argmax()
        
        # Track this token's probability across layers
        trajectory = probs[:, final_top_idx]  # [L]
        
        # Find first layer exceeding threshold
        above_threshold = (trajectory >= threshold).nonzero(as_tuple=True)[0]
        
        if len(above_threshold) == 0:
            return -1
        
        return logit_capture.captured_layers[above_threshold[0].item()]
    
    def entropy_by_layer(
        self,
        logit_capture,
        position: int
    ) -> torch.Tensor:
        """
        Prediction entropy at each layer (uncertainty measure).
        
        Low entropy = confident (peaked distribution)
        High entropy = uncertain (flat distribution)
        
        Args:
            logit_capture: LogitCapture from adapter.capture_logits()
            position: Token position to analyze
            
        Returns:
            [L] entropy values (in nats)
        """
        if logit_capture.logits is None:
            # Can approximate from top-k
            logits = logit_capture.top_k_logits[:, position, :]  # [L, K]
        else:
            logits = logit_capture.logits[:, position, :]  # [L, V]
        
        probs = F.softmax(logits, dim=-1)  # [L, V] or [L, K]
        
        # Entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # [L]
        
        return entropy
    
    def prediction_stability(
        self,
        logit_capture,
        position: int
    ) -> Dict[str, Any]:
        """
        Analyze how stable predictions are across layers.
        
        Args:
            logit_capture: LogitCapture from adapter.capture_logits()
            position: Token position to analyze
            
        Returns:
            Dict with analysis results:
            - 'final_prediction': str
            - 'first_correct_layer': int (when final prediction first appears as top)
            - 'stability': float (fraction of layers with correct top prediction)
            - 'entropy_trend': List[float] (entropy at each layer)
        """
        predictions = self.layer_predictions(logit_capture, position, top_k=1)
        entropy = self.entropy_by_layer(logit_capture, position)
        
        final_pred = predictions[-1]['top_token']
        
        # Find first layer with correct prediction
        first_correct = -1
        correct_count = 0
        for pred in predictions:
            if pred['top_token'] == final_pred:
                correct_count += 1
                if first_correct == -1:
                    first_correct = pred['layer']
        
        return {
            'final_prediction': final_pred,
            'first_correct_layer': first_correct,
            'stability': correct_count / len(predictions),
            'entropy_trend': entropy.tolist(),
        }
