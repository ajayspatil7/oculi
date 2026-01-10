"""
Patching Experiments
====================

High-level API for running systematic patching experiments.

Provides:
    - PatchingContext: Context manager for applying patches during forward pass
    - CausalTracer: Systematic sweeps across layers and components

Usage:
    >>> from oculi.intervention import CausalTracer
    >>> 
    >>> tracer = CausalTracer(adapter)
    >>> results = tracer.trace(
    ...     clean_input=clean_ids,
    ...     corrupted_input=corrupt_ids,
    ...     metric_fn=lambda logits: logits[0, -1, target_token].item(),
    ... )
    >>> 
    >>> print(results.summary())
    >>> heatmap = results.recovery_matrix()
"""

from typing import List, Callable, Optional, Union, Dict, Any, TYPE_CHECKING
from contextlib import contextmanager
import torch

from oculi.intervention.patching import (
    PatchConfig,
    ActivationPatch,
    PatchingResult,
    PatchingSweepResult,
    VALID_COMPONENTS,
)

if TYPE_CHECKING:
    from oculi.models.base import AttentionAdapter


# =============================================================================
# PATCHING CONTEXT
# =============================================================================

class PatchingContext:
    """
    Context manager for applying activation patches during forward pass.
    
    Patches are automatically removed on context exit.
    
    Usage:
        >>> patch = ActivationPatch(
        ...     config=PatchConfig(layer=20, component='mlp_out'),
        ...     source_activation=clean_mlp_output
        ... )
        >>> 
        >>> with PatchingContext(adapter, [patch]):
        ...     output = model(corrupted_ids)
        >>>
        >>> # Patches automatically removed
    
    Multiple patches:
        >>> patches = [
        ...     ActivationPatch(PatchConfig(20, 'mlp_out'), clean_mlp_20),
        ...     ActivationPatch(PatchConfig(21, 'attn_out'), clean_attn_21),
        ... ]
        >>> 
        >>> with PatchingContext(adapter, patches):
        ...     output = model(corrupted_ids)
    """
    
    def __init__(
        self,
        adapter: 'AttentionAdapter',
        patches: List[ActivationPatch],
    ):
        """
        Initialize patching context.
        
        Args:
            adapter: The model adapter
            patches: List of patches to apply
            
        Raises:
            ValueError: If any patch fails validation
        """
        self.adapter = adapter
        self.patches = patches
        self._hook_handles: List = []
        self._entered = False
        
        # Validate all patches upfront
        for patch in patches:
            patch.validate(adapter)
    
    def __enter__(self) -> 'PatchingContext':
        """Apply all patches by registering hooks."""
        if self._entered:
            raise RuntimeError("PatchingContext already entered")
        
        self._entered = True
        
        for patch in self.patches:
            self._apply_patch(patch)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Remove all patches."""
        self._cleanup()
        self._entered = False
        return False  # Don't suppress exceptions
    
    def _apply_patch(self, patch: ActivationPatch) -> None:
        """
        Apply a single patch by registering appropriate hook.
        
        The hook intercepts the activation and replaces it with the source.
        """
        component = patch.component
        layer = patch.layer
        
        # Create the patching hook
        def make_patch_hook(p: ActivationPatch):
            """Factory to capture patch in closure."""
            def hook(module, args, output):
                # Apply patch
                if isinstance(output, tuple):
                    # Some modules return tuples
                    patched = p.apply(output[0].squeeze(0))
                    return (patched.unsqueeze(0),) + output[1:]
                else:
                    patched = p.apply(output.squeeze(0))
                    return patched.unsqueeze(0)
            return hook
        
        hook_fn = make_patch_hook(patch)
        
        # Get the module to hook based on component
        module = self._get_module_for_component(layer, component, patch.head)
        
        if module is not None:
            handle = module.register_forward_hook(hook_fn)
            self._hook_handles.append(handle)
    
    def _get_module_for_component(
        self,
        layer: int,
        component: str,
        head: Optional[int] = None
    ):
        """Get the PyTorch module corresponding to a component."""
        model = self.adapter.model
        
        # Access layer
        try:
            layer_module = model.model.layers[layer]
        except (AttributeError, IndexError):
            return None
        
        if component == 'residual_pre_attn':
            # Hook on input_layernorm (captures block input)
            return layer_module.input_layernorm
        
        elif component == 'residual_post_attn':
            # Hook on self_attn output
            return layer_module.self_attn
        
        elif component == 'residual_post_mlp':
            # Hook on mlp output  
            return layer_module.mlp
        
        elif component == 'attn_out':
            # Hook on attention output projection
            return layer_module.self_attn.o_proj
        
        elif component == 'mlp_out':
            # Hook on MLP down projection
            return layer_module.mlp.down_proj
        
        elif component == 'head':
            # For head-level patching, we need special handling
            # This hooks the full attention output and modifies one head
            return layer_module.self_attn.o_proj
        
        else:
            return None
    
    def _cleanup(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass  # Best effort cleanup
        self._hook_handles.clear()


@contextmanager
def patching_context(
    adapter: 'AttentionAdapter',
    patches: List[ActivationPatch]
):
    """
    Functional form of PatchingContext.
    
    Usage:
        >>> with patching_context(adapter, patches) as ctx:
        ...     output = model(input_ids)
    """
    ctx = PatchingContext(adapter, patches)
    with ctx:
        yield ctx


# =============================================================================
# CAUSAL TRACER
# =============================================================================

class CausalTracer:
    """
    High-level API for systematic patching experiments.
    
    Performs causal tracing by:
    1. Running model on clean input → get clean activations
    2. Running model on corrupted input → get baseline metric
    3. Systematically patching clean activations → sweeping layers/components
    4. Measuring recovery for each patch
    
    Usage:
        >>> tracer = CausalTracer(adapter)
        >>> 
        >>> results = tracer.trace(
        ...     clean_input=clean_ids,
        ...     corrupted_input=corrupt_ids,
        ...     metric_fn=lambda logits: logits[0, -1, target_token].item(),
        ...     layers=range(20, 32),
        ...     components=['mlp_out', 'attn_out'],
        ... )
        >>> 
        >>> print(results.summary())
        >>> heatmap = results.recovery_matrix()
    """
    
    def __init__(self, adapter: 'AttentionAdapter'):
        """
        Initialize CausalTracer.
        
        Args:
            adapter: The model adapter to use
        """
        self.adapter = adapter
        self.model = adapter.model
    
    def trace(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        metric_fn: Callable[[torch.Tensor], float],
        layers: Optional[List[int]] = None,
        components: Optional[List[str]] = None,
        tokens: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> PatchingSweepResult:
        """
        Run systematic patching sweep.
        
        Args:
            clean_input: Input IDs for clean run [1, seq_len]
            corrupted_input: Input IDs for corrupted run [1, seq_len]
            metric_fn: Function that takes logits and returns a scalar metric
                       Example: lambda logits: logits[0, -1, target_id].item()
            layers: Layers to sweep (default: all layers)
            components: Components to sweep (default: ['mlp_out', 'attn_out'])
            tokens: Token positions to patch (default: all)
            verbose: Print progress
            
        Returns:
            PatchingSweepResult containing all results
        """
        # Defaults
        if layers is None:
            layers = list(range(self.adapter.num_layers()))
        
        if components is None:
            components = ['mlp_out', 'attn_out']
        
        # Validate components
        for comp in components:
            if comp not in VALID_COMPONENTS:
                raise ValueError(f"Invalid component: {comp}")
        
        # Step 1: Get clean metric and clean captures
        clean_captures = self._capture_all_components(clean_input, layers, components)
        
        with torch.no_grad():
            clean_logits = self.model(clean_input).logits
            metric_clean = metric_fn(clean_logits)
        
        if verbose:
            print(f"Clean metric: {metric_clean:.4f}")
        
        # Step 2: Get corrupted baseline
        with torch.no_grad():
            corrupted_logits = self.model(corrupted_input).logits
            metric_corrupted = metric_fn(corrupted_logits)
        
        if verbose:
            print(f"Corrupted metric: {metric_corrupted:.4f}")
            print(f"Difference: {metric_clean - metric_corrupted:.4f}")
            print()
        
        # Step 3: Systematic sweep
        results = []
        
        for layer in layers:
            for component in components:
                if verbose:
                    print(f"Patching L{layer} {component}...", end=" ")
                
                # Get source activation for this layer/component
                source = clean_captures.get((layer, component))
                
                if source is None:
                    if verbose:
                        print("skipped (no capture)")
                    continue
                
                # Create patch
                config = PatchConfig(
                    layer=layer,
                    component=component,
                    tokens=tokens,
                )
                patch = ActivationPatch(config=config, source_activation=source)
                
                # Run with patch
                try:
                    with PatchingContext(self.adapter, [patch]):
                        patched_logits = self.model(corrupted_input).logits
                        metric_patched = metric_fn(patched_logits)
                except Exception as e:
                    if verbose:
                        print(f"failed ({e})")
                    continue
                
                # Compute result
                result = PatchingResult.compute(
                    config=config,
                    metric_clean=metric_clean,
                    metric_corrupted=metric_corrupted,
                    metric_patched=metric_patched,
                )
                results.append(result)
                
                if verbose:
                    print(f"recovery={result.recovery:.3f}")
        
        return PatchingSweepResult(
            results=results,
            layers=layers,
            components=components,
        )
    
    def _capture_all_components(
        self,
        input_ids: torch.Tensor,
        layers: List[int],
        components: List[str],
    ) -> Dict[tuple, torch.Tensor]:
        """
        Capture all needed activations from a forward pass.
        
        Returns:
            Dict mapping (layer, component) -> activation tensor
        """
        from oculi import ResidualConfig, MLPConfig
        
        captures = {}
        
        # Determine what we need to capture
        need_residual = any(c.startswith('residual_') for c in components)
        need_mlp = 'mlp_out' in components
        need_attn = 'attn_out' in components
        
        # Capture residual stream
        if need_residual:
            residual_config = ResidualConfig(
                layers=layers,
                capture_pre_attn='residual_pre_attn' in components,
                capture_post_attn='residual_post_attn' in components,
                capture_post_mlp='residual_post_mlp' in components,
            )
            residual = self.adapter.capture_residual(input_ids, residual_config)
            
            for i, layer in enumerate(layers):
                if 'residual_pre_attn' in components and residual.pre_attn is not None:
                    captures[(layer, 'residual_pre_attn')] = residual.pre_attn[i]
                if 'residual_post_attn' in components and residual.post_attn is not None:
                    captures[(layer, 'residual_post_attn')] = residual.post_attn[i]
                if 'residual_post_mlp' in components and residual.post_mlp is not None:
                    captures[(layer, 'residual_post_mlp')] = residual.post_mlp[i]
        
        # Capture MLP output
        if need_mlp:
            mlp_config = MLPConfig(
                layers=layers,
                capture_output=True,
                capture_gate=False,
                capture_pre_activation=False,
                capture_post_activation=False,
            )
            mlp = self.adapter.capture_mlp(input_ids, mlp_config)
            
            for i, layer in enumerate(layers):
                if mlp.mlp_output is not None:
                    captures[(layer, 'mlp_out')] = mlp.mlp_output[i]
        
        # Capture attention output
        if need_attn:
            # For attention output, we need to capture the o_proj output
            # This requires a custom hook since it's not in standard captures
            self._capture_attn_outputs(input_ids, layers, captures)
        
        return captures
    
    def _capture_attn_outputs(
        self,
        input_ids: torch.Tensor,
        layers: List[int],
        captures: Dict[tuple, torch.Tensor],
    ) -> None:
        """Capture attention outputs via hooks."""
        storage = {}
        handles = []
        
        def make_hook(layer_idx):
            def hook(module, args, output):
                # o_proj output: [batch, seq, hidden]
                storage[layer_idx] = output.detach().clone().squeeze(0)
            return hook
        
        try:
            for layer in layers:
                o_proj = self.adapter.model.model.layers[layer].self_attn.o_proj
                handle = o_proj.register_forward_hook(make_hook(layer))
                handles.append(handle)
            
            with torch.no_grad():
                self.adapter.model(input_ids)
            
            for layer in layers:
                if layer in storage:
                    captures[(layer, 'attn_out')] = storage[layer]
        
        finally:
            for h in handles:
                h.remove()
    
    def single_patch(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        metric_fn: Callable[[torch.Tensor], float],
        layer: int,
        component: str,
        head: Optional[int] = None,
        tokens: Optional[List[int]] = None,
    ) -> PatchingResult:
        """
        Run a single patching experiment.
        
        Convenience method for testing a specific hypothesis.
        
        Args:
            clean_input: Clean input IDs
            corrupted_input: Corrupted input IDs
            metric_fn: Metric function
            layer: Layer to patch
            component: Component to patch
            head: Head index (for component='head')
            tokens: Token positions to patch
            
        Returns:
            PatchingResult for this single patch
        """
        results = self.trace(
            clean_input=clean_input,
            corrupted_input=corrupted_input,
            metric_fn=metric_fn,
            layers=[layer],
            components=[component],
            tokens=tokens,
        )
        
        if results.results:
            return results.results[0]
        else:
            raise RuntimeError(f"Patching failed for L{layer} {component}")
