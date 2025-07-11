"""
FLUX attention processor with ToMe (Token Merging) and Regional Dream Renderer capabilities.
"""

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    # Mock torch for environments where it's not available
    torch = type('MockTorch', (), {
        'tensor': lambda x: x,
        'zeros': lambda *args, **kwargs: [[0]],
        'stack': lambda x, **kwargs: x,
        'norm': lambda x: 1.0,
        'exp': lambda x: x,
        'meshgrid': lambda *args, **kwargs: (args[0], args[1]),
        'linspace': lambda *args, **kwargs: [0, 0.5, 1],
        'Tensor': type,
        'FloatTensor': type,
    })()
    F = type('MockF', (), {
        'softmax': lambda x, **kwargs: x,
        'interpolate': lambda x, **kwargs: x,
    })()
    TORCH_AVAILABLE = False

from typing import Optional, Dict, Any, List, Tuple
from tome_flux_utils import (
    get_attention_centroid, 
    compute_pose_loss, 
    flux_attention_aggregation
)


class RegionalDreamRendererFluxAttnProcessor2_0:
    """
    Base attention processor for Regional Dream Renderer FLUX.
    This is a placeholder that would normally extend the actual FLUX attention processor.
    """
    def __init__(self):
        pass
    
    def __call__(self, attn, hidden_states, **kwargs):
        """Placeholder for base regional attention processing."""
        # In actual implementation, this would handle regional attention
        return hidden_states


class AttentionStore:
    """
    Store and manage attention maps during the denoising process.
    Adapted from the original ToMe implementation for FLUX compatibility.
    """
    def __init__(self, attention_res: int = 32, save_global_store: bool = True):
        self.attention_res = attention_res
        self.save_global_store = save_global_store
        self.step_store = []
        self.attention_store = []
        self.curr_step_index = 0
        
    def get_empty_store(self):
        """Return empty attention store structure."""
        return {"down": [], "mid": [], "up": []}
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """
        Store attention maps during forward pass.
        
        Args:
            attn: Attention tensor
            is_cross: Whether this is cross-attention
            place_in_unet: Location in UNet ("down", "mid", "up")
        """
        if attn.shape[-1] <= self.attention_res ** 2:
            if is_cross:
                key = f"{place_in_unet}_{'cross'}"
                self.step_store[self.curr_step_index][key].append(attn)
    
    def between_steps(self):
        """Called between denoising steps."""
        if self.save_global_store:
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for i in range(len(self.attention_store)):
                    for key in self.attention_store[i]:
                        self.attention_store[i][key] += self.step_store[i][key]
        
        self.step_store = []
        self.curr_step_index = 0
    
    def get_average_attention(self):
        """Get averaged attention maps across all stored steps."""
        if len(self.attention_store) == 0:
            return None
            
        average_attention = []
        for store_dict in self.attention_store:
            all_attn = []
            for key in store_dict:
                if len(store_dict[key]) > 0:
                    all_attn.extend(store_dict[key])
            
            if len(all_attn) > 0:
                # Stack and average
                stacked_attn = torch.stack(all_attn, dim=0)
                avg_attn = stacked_attn.mean(dim=0)
                average_attention.append(avg_attn)
        
        if len(average_attention) > 0:
            return torch.stack(average_attention, dim=0).mean(dim=0)
        return None
    
    def reset(self):
        """Reset the attention store."""
        self.step_store = []
        self.attention_store = []
        self.curr_step_index = 0


class ToMeFluxAttnProcessor(RegionalDreamRendererFluxAttnProcessor2_0):
    """
    FLUX attention processor with ToMe (Token Merging) and Regional Dream Renderer capabilities.
    
    This processor extends the regional attention functionality with:
    - Token merging for semantic binding
    - Attention refinement for improved focus
    - Regional control for spatially-aware generation
    """
    
    def __init__(self, attention_res: int = 32):
        super().__init__()
        self.attention_store = None
        self.token_refinement_active = False
        self.attention_refinement_active = False
        self.attention_res = attention_res
        self.regional_masks = {}
        self.regional_prompts = {}
        
    def set_attention_store(self, attention_store: AttentionStore):
        """Set the attention store for tracking attention maps."""
        self.attention_store = attention_store
        
    def enable_token_refinement(self, enable: bool = True):
        """Enable or disable token refinement."""
        self.token_refinement_active = enable
        
    def enable_attention_refinement(self, enable: bool = True):
        """Enable or disable attention refinement."""
        self.attention_refinement_active = enable
        
    def set_regional_config(self, regional_masks: Dict[str, torch.Tensor], 
                           regional_prompts: Dict[str, str]):
        """
        Set regional configuration for spatially-aware attention.
        
        Args:
            regional_masks: Dictionary mapping region names to mask tensors
            regional_prompts: Dictionary mapping region names to prompt strings
        """
        self.regional_masks = regional_masks
        self.regional_prompts = regional_prompts
    
    def store_attention(self, attn: torch.Tensor, hidden_states: torch.Tensor, 
                       layer_name: str = "unknown"):
        """
        Store attention maps for later processing.
        
        Args:
            attn: Attention tensor
            hidden_states: Hidden state tensor
            layer_name: Name of the layer for identification
        """
        if self.attention_store is None:
            return
            
        # Determine if this is cross-attention based on shape
        is_cross = attn.shape[-1] != attn.shape[-2]
        
        # Determine placement in network (simplified)
        place_in_unet = "mid"  # Default placement
        if "down" in layer_name.lower():
            place_in_unet = "down"
        elif "up" in layer_name.lower():
            place_in_unet = "up"
            
        # Store attention if resolution is appropriate
        self.attention_store.forward(attn, is_cross, place_in_unet)
    
    def apply_regional_attention(self, attn: torch.Tensor, 
                               hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply regional attention control to focus on specific spatial areas.
        
        Args:
            attn: Attention tensor of shape [batch, heads, seq_len, seq_len]
            hidden_states: Hidden states tensor
            
        Returns:
            Modified hidden states with regional attention applied
        """
        if len(self.regional_masks) == 0:
            return hidden_states
            
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply regional masks to attention
        for region_name, mask in self.regional_masks.items():
            if mask is not None:
                # Resize mask to match attention resolution
                if mask.dim() == 2:
                    mask_h, mask_w = mask.shape
                    attn_h = attn_w = int(seq_len ** 0.5)  # Assume square attention
                    
                    if mask_h != attn_h or mask_w != attn_w:
                        mask = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0),
                            size=(attn_h, attn_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    
                    # Flatten mask to match sequence length
                    mask_flat = mask.flatten()
                    
                    # Apply mask to attention weights
                    if mask_flat.shape[0] == attn.shape[-1]:
                        mask_expanded = mask_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        attn = attn * mask_expanded
        
        return hidden_states
    
    def apply_tome_refinement(self, hidden_states: torch.Tensor, 
                            attn: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply ToMe attention refinement to improve semantic binding.
        
        Args:
            hidden_states: Hidden states tensor
            attn: Attention tensor
            **kwargs: Additional arguments
            
        Returns:
            Refined hidden states
        """
        if not self.attention_refinement_active:
            return hidden_states
            
        # Get token indices to refine from kwargs
        indices_to_alter = kwargs.get('indices_to_alter', [])
        if len(indices_to_alter) == 0:
            return hidden_states
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply refinement to specific tokens
        for indices_group in indices_to_alter:
            if len(indices_group) == 0 or len(indices_group[0]) == 0:
                continue
                
            primary_idx = indices_group[0][0]
            if primary_idx >= seq_len:
                continue
            
            # Get attention for primary token
            token_attn = attn[:, :, primary_idx, :]  # [batch, heads, seq_len]
            
            # Compute attention centroid for spatial focus
            attn_2d = token_attn.mean(dim=1)  # Average over heads
            
            # Reshape to spatial dimensions (assuming square)
            spatial_size = int(seq_len ** 0.5)
            if spatial_size * spatial_size == seq_len:
                attn_spatial = attn_2d.view(batch_size, spatial_size, spatial_size)
                
                # Compute centroid
                centroids = []
                for b in range(batch_size):
                    centroid = get_attention_centroid(attn_spatial[b])
                    centroids.append(centroid)
                
                # Apply spatial refinement based on centroid
                if len(centroids) > 0:
                    # Enhance attention around centroid
                    for b in range(batch_size):
                        centroid = centroids[b]
                        
                        # Create Gaussian attention enhancement
                        y_grid, x_grid = torch.meshgrid(
                            torch.linspace(0, 1, spatial_size, device=hidden_states.device),
                            torch.linspace(0, 1, spatial_size, device=hidden_states.device),
                            indexing='ij'
                        )
                        
                        # Distance from centroid
                        dist = torch.sqrt((x_grid - centroid[0])**2 + (y_grid - centroid[1])**2)
                        
                        # Gaussian enhancement
                        gaussian = torch.exp(-dist**2 / (2 * 0.1**2))
                        gaussian_flat = gaussian.flatten()
                        
                        # Apply enhancement to hidden states
                        enhancement = gaussian_flat.unsqueeze(-1) * 0.1
                        hidden_states[b] = hidden_states[b] + enhancement
        
        return hidden_states
    
    def compute_attention_statistics(self, attn: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute statistics about attention patterns for analysis and debugging.
        
        Args:
            attn: Attention tensor
            
        Returns:
            Dictionary with attention statistics
        """
        stats = {}
        
        # Attention entropy (measure of focus)
        attn_flat = attn.flatten(2)  # Flatten spatial dimensions
        attn_probs = F.softmax(attn_flat, dim=-1)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1)
        stats['entropy'] = entropy.mean()
        
        # Attention concentration (peak value)
        stats['max_attention'] = attn.max()
        stats['mean_attention'] = attn.mean()
        
        # Spatial spread
        if attn.dim() >= 3:
            spatial_attn = attn.mean(dim=1)  # Average over heads
            stats['spatial_variance'] = spatial_attn.var()
        
        return stats
    
    def __call__(self, attn_module, hidden_states: torch.Tensor, 
                 encoder_hidden_states: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """
        Main forward pass for ToMe FLUX attention processor.
        
        Args:
            attn_module: The attention module
            hidden_states: Input hidden states
            encoder_hidden_states: Encoder hidden states for cross-attention
            attention_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Processed hidden states
        """
        # Store original shape
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply base regional attention processing
        output = super().__call__(attn_module, hidden_states, **kwargs)
        
        # Compute attention weights for analysis and storage
        query = hidden_states
        key = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
        # Simplified attention computation for demonstration
        scale = (hidden_dim // 8) ** -0.5  # Assuming 8 heads
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # Store attention maps if attention store is active
        if self.attention_store is not None:
            layer_name = kwargs.get('layer_name', 'unknown')
            self.store_attention(attn_probs, hidden_states, layer_name)
        
        # Apply regional attention control
        output = self.apply_regional_attention(attn_probs, output)
        
        # Apply ToMe attention refinement if active
        if self.token_refinement_active or self.attention_refinement_active:
            output = self.apply_tome_refinement(output, attn_probs, **kwargs)
        
        return output


class MultiRegionAttentionProcessor(ToMeFluxAttnProcessor):
    """
    Extended attention processor for handling multiple regions with different prompts.
    """
    
    def __init__(self, attention_res: int = 32, max_regions: int = 4):
        super().__init__(attention_res)
        self.max_regions = max_regions
        self.region_attention_weights = {}
        
    def set_region_weights(self, region_weights: Dict[str, float]):
        """
        Set attention weights for different regions.
        
        Args:
            region_weights: Dictionary mapping region names to attention weights
        """
        self.region_attention_weights = region_weights
        
    def apply_multi_region_attention(self, attn: torch.Tensor, 
                                   hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply attention control across multiple regions with different weights.
        
        Args:
            attn: Attention tensor
            hidden_states: Hidden states tensor
            
        Returns:
            Modified hidden states with multi-region attention
        """
        if len(self.regional_masks) == 0:
            return hidden_states
            
        batch_size, seq_len, hidden_dim = hidden_states.shape
        output = hidden_states.clone()
        
        # Apply each regional mask with its weight
        for region_name, mask in self.regional_masks.items():
            if mask is None:
                continue
                
            weight = self.region_attention_weights.get(region_name, 1.0)
            
            # Process this region
            regional_output = self.apply_regional_attention(attn, hidden_states)
            
            # Blend with base output using region weight
            if mask.dim() == 2:
                # Expand mask to match hidden states
                mask_expanded = mask.unsqueeze(0).unsqueeze(-1)
                mask_expanded = F.interpolate(
                    mask_expanded, 
                    size=(seq_len, hidden_dim),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                # Apply weighted blending
                output = output * (1 - weight * mask_expanded) + \
                        regional_output * (weight * mask_expanded)
        
        return output
    
    def __call__(self, attn_module, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Override to use multi-region processing."""
        # Store original shape
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply base regional attention processing
        output = RegionalDreamRendererFluxAttnProcessor2_0.__call__(
            self, attn_module, hidden_states, **kwargs
        )
        
        # Compute attention weights
        query = hidden_states
        key = kwargs.get('encoder_hidden_states', hidden_states)
        
        scale = (hidden_dim // 8) ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # Store attention if needed
        if self.attention_store is not None:
            layer_name = kwargs.get('layer_name', 'unknown')
            self.store_attention(attn_probs, hidden_states, layer_name)
        
        # Apply multi-region attention control
        output = self.apply_multi_region_attention(attn_probs, output)
        
        # Apply ToMe refinement
        if self.token_refinement_active or self.attention_refinement_active:
            output = self.apply_tome_refinement(output, attn_probs, **kwargs)
        
        return output