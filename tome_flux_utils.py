"""
FLUX-adapted utility functions for ToMe integration.
Adapts existing ToMe utilities for FLUX's transformer architecture.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import numpy as np


def flux_token_merge(prompt_embeds: torch.Tensor, 
                    idx_merge: List[List[int]], 
                    merge_strategy: str = "weighted_sum") -> torch.Tensor:
    """
    FLUX-adapted token merging function
    
    Args:
        prompt_embeds: [batch_size, seq_len, hidden_dim] - FLUX token embeddings
        idx_merge: List of token indices to merge, e.g., [[[1],[2]],[[3],[4]]]
        merge_strategy: "weighted_sum", "attention_weighted", or "learnable"
    
    Returns:
        torch.Tensor: Merged token embeddings
    """
    for idxs in idx_merge:
        if len(idxs) >= 2:
            noun_idx = idxs[0][0]
            if merge_strategy == "weighted_sum":
                # Adaptive weighting based on token importance
                alpha, beta = 1.1, 1.2
                prompt_embeds[:, noun_idx] = (
                    alpha * prompt_embeds[:, idxs[0]].sum(dim=1) + 
                    beta * prompt_embeds[:, idxs[1]].sum(dim=1)
                )
            elif merge_strategy == "attention_weighted":
                # Use attention scores for weighting (placeholder for now)
                weights = compute_token_importance(prompt_embeds, idxs)
                prompt_embeds[:, noun_idx] = torch.sum(
                    prompt_embeds[:, idxs[0] + idxs[1]] * weights.unsqueeze(-1), 
                    dim=1
                )
            
            # Zero out merged tokens
            if len(idxs[0]) > 1:
                prompt_embeds[:, idxs[0][1:]] = 0
            prompt_embeds[:, idxs[1]] = 0
    
    return prompt_embeds


def compute_token_importance(prompt_embeds: torch.Tensor, 
                           token_indices: List[List[int]]) -> torch.Tensor:
    """
    Compute importance weights for tokens based on their embeddings.
    
    Args:
        prompt_embeds: Token embeddings
        token_indices: Indices of tokens to compute weights for
    
    Returns:
        torch.Tensor: Importance weights
    """
    all_indices = token_indices[0] + token_indices[1]
    token_embeds = prompt_embeds[:, all_indices]  # [batch_size, num_tokens, hidden_dim]
    
    # Compute attention-like scores based on embedding magnitudes
    scores = torch.norm(token_embeds, dim=-1)  # [batch_size, num_tokens]
    weights = F.softmax(scores / 0.1, dim=-1)  # Temperature-scaled softmax
    
    return weights


def flux_semantic_binding_loss(transformer_output: torch.Tensor,
                              target_embeds: torch.Tensor, 
                              region_mask: Optional[torch.Tensor] = None,
                              temperature: float = 0.5) -> torch.Tensor:
    """
    Compute semantic binding loss for FLUX transformer
    
    Args:
        transformer_output: Output from FLUX transformer [batch_size, seq_len, hidden_dim]
        target_embeds: Target embeddings for binding [batch_size, seq_len, hidden_dim]
        region_mask: Optional regional mask for localized loss [batch_size, h, w]
        temperature: Temperature for attention computation
    
    Returns:
        torch.Tensor: Semantic binding loss
    """
    if region_mask is not None:
        # Apply regional mask to focus loss on specific areas
        # Expand mask to match embedding dimensions
        if len(region_mask.shape) == 3:  # [batch_size, h, w]
            # Reshape to match transformer output if needed
            batch_size, h, w = region_mask.shape
            region_mask = region_mask.view(batch_size, -1)  # [batch_size, h*w]
            
        # Apply mask to both outputs and targets
        masked_output = transformer_output * region_mask.unsqueeze(-1)
        masked_target = target_embeds * region_mask.unsqueeze(-1)
        loss = F.mse_loss(masked_output, masked_target)
    else:
        loss = F.mse_loss(transformer_output, target_embeds)
    
    return loss


def flux_attention_entropy_loss(attention_maps: torch.Tensor,
                               indices_to_alter: List[int],
                               temperature: float = 0.5) -> torch.Tensor:
    """
    Compute entropy-based attention loss for FLUX
    
    Args:
        attention_maps: Attention maps from FLUX transformer [batch_size, heads, seq_len, seq_len]
        indices_to_alter: Token indices to compute entropy loss for
        temperature: Temperature for softmax computation
    
    Returns:
        torch.Tensor: Entropy loss
    """
    batch_size, num_heads, seq_len, _ = attention_maps.shape
    
    # Focus on cross-attention to specific tokens
    target_attention = attention_maps[:, :, :, indices_to_alter]  # [batch, heads, seq_len, num_targets]
    
    # Apply softmax with temperature
    attention_probs = F.softmax(target_attention / temperature, dim=-1)
    
    # Compute entropy loss (negative entropy to encourage sharpness)
    entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
    loss = -entropy.mean()  # Negative to encourage low entropy (sharp attention)
    
    return loss


def regional_token_merge(regional_embeds: List[torch.Tensor], 
                        regional_masks: List[torch.Tensor],
                        indices_to_alter: List[List[List[int]]]) -> List[torch.Tensor]:
    """
    Apply token merging per region with cross-regional binding
    
    Args:
        regional_embeds: List of embeddings for each region
        regional_masks: List of masks defining regions
        indices_to_alter: Token indices to merge per region
    
    Returns:
        List[torch.Tensor]: Enhanced embeddings per region
    """
    enhanced_embeds = []
    
    for i, (embeds, mask) in enumerate(zip(regional_embeds, regional_masks)):
        # Apply region-specific token merging
        merged_embeds = flux_token_merge(embeds, indices_to_alter[i])
        
        # Apply cross-regional semantic binding if not the first region
        if i > 0:
            # Bind with previous regions for consistency
            binding_loss = cross_regional_binding_loss(
                merged_embeds, enhanced_embeds[-1], mask
            )
            # Optimize embeddings based on binding loss (simplified)
            merged_embeds = optimize_cross_binding(merged_embeds, binding_loss)
            
        enhanced_embeds.append(merged_embeds)
    
    return enhanced_embeds


def cross_regional_binding_loss(current_embeds: torch.Tensor,
                               previous_embeds: torch.Tensor,
                               current_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute loss for cross-regional semantic consistency
    
    Args:
        current_embeds: Current region embeddings
        previous_embeds: Previous region embeddings
        current_mask: Mask for current region
    
    Returns:
        torch.Tensor: Cross-regional binding loss
    """
    # Compute similarity between current and previous embeddings
    similarity = F.cosine_similarity(current_embeds, previous_embeds, dim=-1)
    
    # Weight by mask importance
    mask_weight = current_mask.sum() / current_mask.numel()
    loss = (1 - similarity).mean() * mask_weight
    
    return loss


def optimize_cross_binding(embeddings: torch.Tensor, 
                          binding_loss: torch.Tensor,
                          learning_rate: float = 0.01) -> torch.Tensor:
    """
    Optimize embeddings based on cross-regional binding loss
    
    Args:
        embeddings: Embeddings to optimize
        binding_loss: Binding loss to minimize
        learning_rate: Learning rate for optimization
    
    Returns:
        torch.Tensor: Optimized embeddings
    """
    # Simple gradient-based update (placeholder implementation)
    if embeddings.requires_grad:
        grad = torch.autograd.grad(binding_loss, embeddings, retain_graph=True)[0]
        optimized_embeds = embeddings - learning_rate * grad
    else:
        # If no gradients, return original embeddings
        optimized_embeds = embeddings
    
    return optimized_embeds


def aggregate_flux_attention(attention_store: Dict[str, torch.Tensor],
                           res: int = 32,
                           select: int = 0) -> torch.Tensor:
    """
    Aggregate attention maps from FLUX transformer blocks
    
    Args:
        attention_store: Dictionary storing attention maps from different layers
        res: Target resolution for attention maps
        select: Which attention head/layer to select
    
    Returns:
        torch.Tensor: Aggregated attention maps [h, w, seq_len]
    """
    attention_maps = []
    
    for layer_name, attn_tensor in attention_store.items():
        if 'transformer' in layer_name.lower():
            # Process FLUX transformer attention
            if len(attn_tensor.shape) == 4:  # [batch, heads, seq_len, seq_len]
                # Select specific head or average across heads
                if select < attn_tensor.shape[1]:
                    attn = attn_tensor[0, select]  # [seq_len, seq_len]
                else:
                    attn = attn_tensor[0].mean(dim=0)  # Average across heads
                attention_maps.append(attn)
    
    if attention_maps:
        # Aggregate across layers
        aggregated = torch.stack(attention_maps).mean(dim=0)
        
        # Reshape to spatial dimensions if needed
        seq_len = aggregated.shape[-1]
        spatial_size = int(np.sqrt(seq_len - 1))  # -1 for CLS token
        
        if spatial_size * spatial_size == seq_len - 1:
            # Reshape excluding CLS token
            spatial_attn = aggregated[1:, 1:].reshape(spatial_size, spatial_size, seq_len)
            # Resize to target resolution
            if spatial_size != res:
                spatial_attn = F.interpolate(
                    spatial_attn.permute(2, 0, 1).unsqueeze(0),
                    size=(res, res),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)
            return spatial_attn
    
    # Fallback: return identity attention map
    return torch.eye(res, device=attention_maps[0].device if attention_maps else 'cpu').unsqueeze(-1)