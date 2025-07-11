"""
Enhanced attention processors for ToMe-FLUX integration.
Extends regional attention processing with ToMe capabilities.
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tome_flux_utils import flux_attention_entropy_loss


class BaseFluxAttnProcessor:
    """Base attention processor for FLUX transformer"""
    
    def __init__(self):
        self.attention_store = None
        self.step_index = 0
        
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        """Base attention computation"""
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        value = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Store attention if needed
        if self.attention_store is not None:
            self._store_attention(attention_probs, attn, hidden_states)
        
        return hidden_states
    
    def _store_attention(self, attention_probs, attn, hidden_states):
        """Store attention maps for later processing"""
        layer_name = f"transformer_block_{self.step_index}"
        if layer_name not in self.attention_store:
            self.attention_store[layer_name] = []
        self.attention_store[layer_name].append(attention_probs.detach())


class RegionalDreamRendererFluxAttnProcessor2_0(BaseFluxAttnProcessor):
    """Regional attention processor for FLUX (base implementation)"""
    
    def __init__(self, regional_prompts: List[str] = None, regional_masks: List[torch.Tensor] = None):
        super().__init__()
        self.regional_prompts = regional_prompts or []
        self.regional_masks = regional_masks or []
        
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        # Apply regional control if masks are available
        if self.regional_masks:
            hidden_states = self._apply_regional_control(attn, hidden_states, encoder_hidden_states, **kwargs)
        else:
            hidden_states = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)
        
        return hidden_states
    
    def _apply_regional_control(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        """Apply regional control to attention computation"""
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Process each region separately
        regional_outputs = []
        for i, mask in enumerate(self.regional_masks):
            # Apply mask to hidden states
            masked_hidden_states = hidden_states * mask.unsqueeze(-1)
            
            # Compute attention for this region
            region_output = super().__call__(attn, masked_hidden_states, encoder_hidden_states, **kwargs)
            regional_outputs.append(region_output)
        
        # Combine regional outputs
        if regional_outputs:
            # Weight by mask coverage
            weights = [mask.sum() / mask.numel() for mask in self.regional_masks]
            weights = torch.tensor(weights, device=hidden_states.device)
            weights = weights / weights.sum()
            
            combined_output = sum(w * output for w, output in zip(weights, regional_outputs))
        else:
            combined_output = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)
            
        return combined_output


class ToMeFluxAttnProcessor(RegionalDreamRendererFluxAttnProcessor2_0):
    """Enhanced attention processor with ToMe capabilities"""
    
    def __init__(self, regional_prompts: List[str] = None, regional_masks: List[torch.Tensor] = None):
        super().__init__(regional_prompts, regional_masks)
        self.token_refinement_active = False
        self.attention_refinement_active = False
        self.tome_config = None
        
    def configure_tome(self, tome_config: Dict[str, Any]):
        """Configure ToMe parameters"""
        self.tome_config = tome_config
        
    def set_token_refinement(self, active: bool):
        """Enable/disable token refinement"""
        self.token_refinement_active = active
        
    def set_attention_refinement(self, active: bool):
        """Enable/disable attention refinement"""
        self.attention_refinement_active = active
        
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        # Apply regional attention as before
        output = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)
        
        # Apply ToMe refinements if active
        if self.token_refinement_active and self.tome_config:
            output = self._apply_token_refinement(output, attn, **kwargs)
            
        if self.attention_refinement_active and self.tome_config:
            output = self._apply_attention_refinement(output, attn, **kwargs)
            
        return output
    
    def _apply_token_refinement(self, hidden_states: torch.Tensor, attn, **kwargs) -> torch.Tensor:
        """Apply ToMe token refinement to FLUX attention output"""
        if not self.tome_config:
            return hidden_states
            
        # Apply token merging if indices are specified
        if 'token_indices' in self.tome_config:
            from tome_flux_utils import flux_token_merge
            indices = self.tome_config['token_indices']
            refined_states = flux_token_merge(hidden_states, indices)
        else:
            refined_states = hidden_states
            
        return refined_states
    
    def _apply_attention_refinement(self, hidden_states: torch.Tensor, attn, **kwargs) -> torch.Tensor:
        """Apply entropy-based attention refinement"""
        if not self.tome_config or self.attention_store is None:
            return hidden_states
            
        # Get stored attention maps
        layer_name = f"transformer_block_{self.step_index}"
        if layer_name in self.attention_store:
            attention_maps = self.attention_store[layer_name][-1]  # Get latest attention
            
            # Compute entropy loss
            if 'indices_to_alter' in self.tome_config:
                indices = self.tome_config['indices_to_alter']
                entropy_loss = flux_attention_entropy_loss(
                    attention_maps.unsqueeze(0), 
                    indices,
                    temperature=self.tome_config.get('temperature', 0.5)
                )
                
                # Apply gradient-based refinement (simplified)
                if hidden_states.requires_grad:
                    grad = torch.autograd.grad(
                        entropy_loss, 
                        hidden_states, 
                        retain_graph=True
                    )[0]
                    
                    scale_factor = self.tome_config.get('scale_factor', 0.1)
                    refined_states = hidden_states - scale_factor * grad
                else:
                    refined_states = hidden_states
            else:
                refined_states = hidden_states
        else:
            refined_states = hidden_states
            
        return refined_states


class FluxAttentionStore:
    """Store and manage attention maps from FLUX transformer"""
    
    def __init__(self, store_self_attention: bool = True, store_cross_attention: bool = True):
        self.store_self_attention = store_self_attention
        self.store_cross_attention = store_cross_attention
        self.self_attns = {}
        self.cross_attns = {}
        self.step_store = {}
        
    def clear(self):
        """Clear stored attention maps"""
        self.self_attns.clear()
        self.cross_attns.clear()
        self.step_store.clear()
        
    def store_attention(self, attention_probs: torch.Tensor, 
                       layer_name: str, 
                       is_cross: bool = False):
        """Store attention maps"""
        if is_cross and self.store_cross_attention:
            if layer_name not in self.cross_attns:
                self.cross_attns[layer_name] = []
            self.cross_attns[layer_name].append(attention_probs.detach().cpu())
        elif not is_cross and self.store_self_attention:
            if layer_name not in self.self_attns:
                self.self_attns[layer_name] = []
            self.self_attns[layer_name].append(attention_probs.detach().cpu())
            
    def get_attention_maps(self, layer_names: Optional[List[str]] = None, 
                          is_cross: bool = False) -> Dict[str, torch.Tensor]:
        """Get stored attention maps"""
        store = self.cross_attns if is_cross else self.self_attns
        
        if layer_names is None:
            return store
        else:
            return {name: store[name] for name in layer_names if name in store}
            
    def aggregate_attention(self, res: int = 32, 
                           from_layers: Optional[List[str]] = None,
                           is_cross: bool = True) -> torch.Tensor:
        """Aggregate attention maps across layers"""
        from tome_flux_utils import aggregate_flux_attention
        
        store = self.cross_attns if is_cross else self.self_attns
        if from_layers:
            attention_store = {name: store[name] for name in from_layers if name in store}
        else:
            attention_store = store
            
        return aggregate_flux_attention(attention_store, res=res)


def register_flux_attention_processors(pipeline, 
                                     regional_prompts: List[str] = None,
                                     regional_masks: List[torch.Tensor] = None,
                                     use_tome: bool = False,
                                     tome_config: Dict[str, Any] = None):
    """Register attention processors for FLUX pipeline"""
    
    if use_tome:
        processor = ToMeFluxAttnProcessor(regional_prompts, regional_masks)
        if tome_config:
            processor.configure_tome(tome_config)
    else:
        processor = RegionalDreamRendererFluxAttnProcessor2_0(regional_prompts, regional_masks)
    
    # Set attention store if needed
    attention_store = FluxAttentionStore()
    processor.attention_store = attention_store
    
    # Register processor to transformer blocks
    if hasattr(pipeline, 'transformer'):
        # Register to all transformer blocks
        for block in pipeline.transformer.transformer_blocks:
            if hasattr(block, 'attn1'):
                block.attn1.set_processor(processor)
            if hasattr(block, 'attn2'):
                block.attn2.set_processor(processor)
    
    return processor, attention_store