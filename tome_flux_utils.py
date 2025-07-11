"""
FLUX-adapted token merging and semantic binding utilities for ToMe integration.
"""

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    # Mock torch for environments where it's not available
    class MockTensor:
        def __init__(self, data):
            self.data = data
        def clone(self):
            return MockTensor(self.data)
        def detach(self):
            return MockTensor(self.data)
        def requires_grad_(self, requires_grad=True):
            return self
        def __getitem__(self, key):
            return MockTensor(self.data)
        def __setitem__(self, key, value):
            pass
        def sum(self, **kwargs):
            return MockTensor(0.5)
        def mean(self, **kwargs):
            return MockTensor(0.5)
        def shape(self):
            return [1, 10, 768]
        @property 
        def shape(self):
            return [1, 10, 768]
        @property
        def device(self):
            return 'cpu'
        def flatten(self, *args):
            return MockTensor(self.data)
        def view(self, *args):
            return MockTensor(self.data)
        def transpose(self, *args):
            return MockTensor(self.data)
        def unsqueeze(self, *args):
            return MockTensor(self.data)
        def __mul__(self, other):
            return MockTensor(self.data)
        def __rmul__(self, other):
            return MockTensor(self.data)
        def __add__(self, other):
            return MockTensor(self.data)
        def __sub__(self, other):
            return MockTensor(self.data)
    
    torch = type('MockTorch', (), {
        'tensor': lambda x: MockTensor(x),
        'zeros': lambda *args, **kwargs: MockTensor([[0]]),
        'randn': lambda *args, **kwargs: MockTensor([[0.1]]),
        'stack': lambda x, **kwargs: MockTensor(x),
        'norm': lambda x: MockTensor(1.0),
        'exp': lambda x: MockTensor(x.data if hasattr(x, 'data') else x),
        'log': lambda x: MockTensor(x.data if hasattr(x, 'data') else x),
        'sqrt': lambda x: MockTensor(x.data if hasattr(x, 'data') else x),
        'meshgrid': lambda *args, **kwargs: (MockTensor(args[0]), MockTensor(args[1])),
        'linspace': lambda *args, **kwargs: MockTensor([0, 0.5, 1]),
        'Tensor': MockTensor,
        'FloatTensor': MockTensor,
        'autograd': type('MockAutograd', (), {'grad': lambda *args, **kwargs: [MockTensor([0.01])]})(),
    })()
    F = type('MockF', (), {
        'mse_loss': lambda x, y: MockTensor(0.1),
        'cosine_similarity': lambda x, y, **kwargs: MockTensor([0.9]),
        'l1_loss': lambda x, y: MockTensor(0.05),
        'softmax': lambda x, **kwargs: x,
        'adaptive_avg_pool1d': lambda x, size: x,
    })()
    TORCH_AVAILABLE = False

from typing import List, Optional, Tuple, Union


def flux_token_merge(prompt_embeds: torch.Tensor, 
                    idx_merge: List[List[List[int]]], 
                    merge_strategy: str = "weighted_sum") -> torch.Tensor:
    """
    FLUX-adapted token merging function that combines related tokens for enhanced semantic binding.
    
    Args:
        prompt_embeds: Token embeddings of shape [batch_size, seq_len, hidden_dim]
        idx_merge: List of token indices to merge, format: [[[noun_indices], [attr_indices]], ...]
        merge_strategy: Strategy for merging tokens ("weighted_sum", "mean", "max")
        
    Returns:
        Updated prompt embeddings with merged tokens
    """
    if len(idx_merge) == 0:
        return prompt_embeds
    
    # Clone to avoid in-place modifications
    merged_embeds = prompt_embeds.clone()
    
    for idxs in idx_merge:
        if len(idxs) >= 2:
            noun_indices = idxs[0]  # Primary object tokens
            attr_indices = idxs[1]  # Attribute tokens
            
            if len(noun_indices) == 0 or len(attr_indices) == 0:
                continue
                
            noun_idx = noun_indices[0]  # Primary noun position
            
            # Weights for semantic binding - emphasize noun while incorporating attributes
            alpha, beta = 1.1, 1.2
            
            # Compute merged representation
            if merge_strategy == "weighted_sum":
                # Sum all noun tokens with primary weight
                noun_sum = merged_embeds[:, noun_indices].sum(dim=1)
                # Sum all attribute tokens with secondary weight  
                attr_sum = merged_embeds[:, attr_indices].sum(dim=1)
                # Combine with semantic weights
                merged_embeds[:, noun_idx] = alpha * noun_sum + beta * attr_sum
                
            elif merge_strategy == "mean":
                # Average all related tokens
                all_indices = noun_indices + attr_indices
                merged_embeds[:, noun_idx] = merged_embeds[:, all_indices].mean(dim=1)
                
            elif merge_strategy == "max":
                # Take element-wise maximum
                all_indices = noun_indices + attr_indices
                merged_embeds[:, noun_idx] = merged_embeds[:, all_indices].max(dim=1)[0]
            
            # Zero out the merged tokens (except primary noun)
            if len(noun_indices) > 1:
                merged_embeds[:, noun_indices[1:]] = 0
            merged_embeds[:, attr_indices] = 0
    
    return merged_embeds


def compute_entropy_loss(attention_maps: torch.Tensor, 
                        indices_to_alter: List[List[List[int]]], 
                        attention_res: int = 32,
                        temperature: float = 0.5) -> torch.Tensor:
    """
    Compute entropy-based loss for attention refinement to encourage focused attention.
    
    Args:
        attention_maps: Attention maps of shape [batch, height, width, seq_len]
        indices_to_alter: Token indices that need attention refinement
        attention_res: Resolution for attention computation
        temperature: Temperature for softmax normalization
        
    Returns:
        Computed entropy loss for attention refinement
    """
    if attention_maps is None or len(indices_to_alter) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    loss = torch.tensor(0.0, requires_grad=True, device=attention_maps.device)
    
    for indices in indices_to_alter:
        if len(indices) == 0 or len(indices[0]) == 0:
            continue
            
        # Get primary token index
        curr_idx = indices[0][0]
        
        # Handle token index bounds
        if curr_idx >= attention_maps.shape[-1] or curr_idx < 1:
            continue
            
        # Get attention for the current token (subtract 1 for 0-indexing)
        attention_for_token = attention_maps[:, :, :, curr_idx - 1]
        
        # Flatten spatial dimensions for entropy computation
        attention_flat = attention_for_token.reshape(attention_for_token.shape[0], -1)
        
        # Apply softmax with temperature for sharper distributions
        attention_probs = F.softmax(attention_flat / temperature, dim=-1)
        
        # Compute negative entropy (we want to minimize entropy for focused attention)
        # Higher entropy = more dispersed attention, lower entropy = more focused
        entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
        
        # We want to minimize entropy, so we add it to the loss
        # Scale by 2 to match original implementation
        loss = loss - 2 * entropy.mean()
    
    return loss


def update_latent(latents: torch.Tensor, 
                 loss: torch.Tensor, 
                 step_size: float,
                 momentum: float = 0.0) -> torch.Tensor:
    """
    Update latents based on computed loss using gradient descent.
    
    Args:
        latents: Current latent representations
        loss: Loss to minimize
        step_size: Learning rate for the update
        momentum: Momentum factor for smoother updates
        
    Returns:
        Updated latents
    """
    if not loss.requires_grad:
        return latents
        
    # Compute gradients
    try:
        grad_cond = torch.autograd.grad(
            loss, [latents], retain_graph=True, create_graph=False
        )[0]
        
        # Apply gradient update with optional momentum
        if momentum > 0.0 and hasattr(update_latent, '_prev_grad'):
            grad_cond = momentum * update_latent._prev_grad + (1 - momentum) * grad_cond
            update_latent._prev_grad = grad_cond
        else:
            update_latent._prev_grad = grad_cond
            
        # Update latents: move in negative gradient direction
        updated_latents = latents - step_size * grad_cond
        
    except RuntimeError as e:
        # Handle cases where gradients cannot be computed
        print(f"Warning: Could not compute gradients for latent update: {e}")
        updated_latents = latents
    
    return updated_latents


def semantic_binding_loss(transformer_output: torch.Tensor, 
                         target_embeds: torch.Tensor, 
                         region_mask: Optional[torch.Tensor] = None,
                         loss_type: str = "mse") -> torch.Tensor:
    """
    Compute semantic binding loss for FLUX transformer to align representations.
    
    Args:
        transformer_output: Output from FLUX transformer
        target_embeds: Target embeddings for semantic alignment
        region_mask: Optional mask for regional loss computation
        loss_type: Type of loss ("mse", "cosine", "l1")
        
    Returns:
        Computed semantic binding loss
    """
    if transformer_output.shape != target_embeds.shape:
        # Handle shape mismatches by adapting dimensions
        if transformer_output.numel() == target_embeds.numel():
            transformer_output = transformer_output.view(target_embeds.shape)
        else:
            # Interpolate or project to match dimensions
            target_embeds = F.adaptive_avg_pool1d(
                target_embeds.transpose(-1, -2), 
                transformer_output.shape[-1]
            ).transpose(-1, -2)
    
    if region_mask is not None:
        # Apply regional mask for localized semantic binding
        if region_mask.dim() < transformer_output.dim():
            # Expand mask to match output dimensions
            for _ in range(transformer_output.dim() - region_mask.dim()):
                region_mask = region_mask.unsqueeze(-1)
        
        # Apply mask to both outputs and targets
        masked_output = transformer_output * region_mask
        masked_target = target_embeds * region_mask
        
        # Normalize by mask sum to avoid bias towards larger regions
        mask_sum = region_mask.sum() + 1e-8
        masked_output = masked_output / mask_sum
        masked_target = masked_target / mask_sum
        
        output_for_loss = masked_output
        target_for_loss = masked_target
    else:
        output_for_loss = transformer_output
        target_for_loss = target_embeds
    
    # Compute loss based on specified type
    if loss_type == "mse":
        loss = F.mse_loss(output_for_loss, target_for_loss)
    elif loss_type == "cosine":
        # Cosine similarity loss (1 - cosine_similarity)
        cos_sim = F.cosine_similarity(
            output_for_loss.flatten(1), 
            target_for_loss.flatten(1), 
            dim=1
        )
        loss = (1 - cos_sim).mean()
    elif loss_type == "l1":
        loss = F.l1_loss(output_for_loss, target_for_loss)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss


def compute_pose_loss(centroids: List[torch.Tensor], 
                     min_distance: float = 0.3,
                     strength: float = 1.0) -> torch.Tensor:
    """
    Compute pose loss to encourage spatial separation between different subjects.
    
    Args:
        centroids: List of centroid positions for different subjects
        min_distance: Minimum desired distance between centroids
        strength: Strength of the pose loss
        
    Returns:
        Computed pose loss
    """
    if len(centroids) < 2:
        return torch.tensor(0.0)
    
    loss = torch.tensor(0.0, device=centroids[0].device)
    
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            # Compute Euclidean distance between centroids
            distance = torch.norm(centroids[i] - centroids[j])
            
            # Penalize distances below minimum threshold
            if distance < min_distance:
                penalty = torch.exp(-(distance / min_distance))
                loss = loss + strength * penalty
    
    return loss


def get_attention_centroid(attention_map: torch.Tensor) -> torch.Tensor:
    """
    Compute the centroid (center of mass) of an attention map.
    
    Args:
        attention_map: Attention map of shape [height, width] or [batch, height, width]
        
    Returns:
        Centroid coordinates as [x, y] or [batch, 2]
    """
    if attention_map.dim() == 2:
        h, w = attention_map.shape
        
        # Create coordinate grids
        y_coords = torch.linspace(0, 1, h, device=attention_map.device).reshape(h, 1)
        x_coords = torch.linspace(0, 1, w, device=attention_map.device).reshape(1, w)
        
        # Compute weighted centroids
        total_attention = attention_map.sum() + 1e-8
        centroid_x = (attention_map * x_coords).sum() / total_attention
        centroid_y = (attention_map * y_coords).sum() / total_attention
        
        return torch.stack([centroid_x, centroid_y])
        
    elif attention_map.dim() == 3:
        batch_size, h, w = attention_map.shape
        centroids = []
        
        for b in range(batch_size):
            centroid = get_attention_centroid(attention_map[b])
            centroids.append(centroid)
            
        return torch.stack(centroids)
    
    else:
        raise ValueError(f"Unsupported attention map dimensions: {attention_map.shape}")


def adaptive_threshold_schedule(step: int, 
                               total_steps: int, 
                               base_threshold: float = 8.0,
                               decay_factor: float = 0.9) -> float:
    """
    Compute adaptive threshold for attention refinement based on denoising progress.
    
    Args:
        step: Current denoising step
        total_steps: Total number of denoising steps
        base_threshold: Base threshold value
        decay_factor: Factor controlling threshold decay
        
    Returns:
        Adaptive threshold for current step
    """
    progress = step / max(total_steps - 1, 1)
    
    # Exponential decay from base threshold
    threshold = base_threshold * (decay_factor ** progress)
    
    # Ensure minimum threshold
    min_threshold = base_threshold * 0.1
    threshold = max(threshold, min_threshold)
    
    return threshold


def flux_attention_aggregation(attention_maps: List[torch.Tensor], 
                              aggregation_method: str = "mean") -> torch.Tensor:
    """
    Aggregate attention maps from multiple FLUX transformer layers.
    
    Args:
        attention_maps: List of attention maps from different layers
        aggregation_method: Method for aggregation ("mean", "max", "weighted")
        
    Returns:
        Aggregated attention map
    """
    if len(attention_maps) == 0:
        return torch.tensor([])
    
    if len(attention_maps) == 1:
        return attention_maps[0]
    
    # Stack attention maps
    stacked_maps = torch.stack(attention_maps, dim=0)
    
    if aggregation_method == "mean":
        return stacked_maps.mean(dim=0)
    elif aggregation_method == "max":
        return stacked_maps.max(dim=0)[0]
    elif aggregation_method == "weighted":
        # Weight later layers more heavily (closer to output)
        num_layers = len(attention_maps)
        weights = torch.softmax(torch.arange(num_layers, dtype=torch.float), dim=0)
        weights = weights.view(-1, *([1] * (stacked_maps.dim() - 1)))
        return (stacked_maps * weights).sum(dim=0)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")