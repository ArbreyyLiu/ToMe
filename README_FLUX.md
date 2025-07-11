# ToMe-FLUX Integration

This directory contains the implementation of ToMe (Token Merging) integration with FLUX Regional Dream Renderer pipeline for enhanced semantic binding in regional text-to-image generation.

## Overview

The ToMe-FLUX integration combines the semantic binding capabilities of Token Merging with FLUX's transformer-based diffusion model to achieve better object-attribute associations and regional consistency in generated images.

### Key Features

- **Regional Control**: Different prompts for different image regions
- **Semantic Binding**: Enhanced object-attribute associations through token merging
- **Cross-Regional Consistency**: Maintain semantic consistency across different regions
- **Training-Free**: No additional training required
- **Flexible Configuration**: Multiple presets for different use cases

## File Structure

```
tome-flux/
├── tome_flux_pipeline.py          # Main ToMe-FLUX pipeline implementation
├── tome_flux_attention.py         # Enhanced attention processors
├── tome_flux_utils.py             # FLUX-adapted utility functions
├── configs/
│   └── tome_flux_config.py        # Configuration classes
├── demo_tome_flux.py              # Demo script and examples
└── README_FLUX.md                 # This documentation
```

## Core Components

### 1. ToMe-FLUX Pipeline (`tome_flux_pipeline.py`)

The main pipeline that extends `RegionalDreamRendererFluxPipeline` with ToMe capabilities:

```python
from tome_flux_pipeline import ToMeFluxPipeline

# Initialize pipeline
pipeline = ToMeFluxPipeline()

# Configure ToMe parameters
pipeline.configure_tome(
    tome_control_steps=[5, 5],
    token_refinement_steps=3,
    attention_refinement_steps=[4, 4],
    scale_factor=20.0
)

# Generate with regional control
result = pipeline(
    prompt="a red cat wearing blue sunglasses and a yellow dog wearing green hat",
    regional_prompts=[
        "a red cat wearing blue sunglasses",
        "a yellow dog wearing green hat"
    ],
    indices_to_alter=[[[2], [3, 4]], [[7], [8, 9]]],
    prompt_merged="a cat and a dog"
)
```

### 2. Enhanced Attention Processors (`tome_flux_attention.py`)

Attention processors that combine regional control with ToMe refinement:

- `ToMeFluxAttnProcessor`: Main processor with ToMe capabilities
- `FluxAttentionStore`: Manages attention map storage and aggregation
- Regional attention control with cross-regional binding

### 3. FLUX-Adapted Utilities (`tome_flux_utils.py`)

Utility functions adapted for FLUX's transformer architecture:

- `flux_token_merge()`: Token merging for FLUX embeddings
- `flux_semantic_binding_loss()`: Semantic binding loss computation
- `regional_token_merge()`: Cross-regional token merging
- `flux_attention_entropy_loss()`: Entropy-based attention refinement

### 4. Configuration System (`configs/tome_flux_config.py`)

Pre-defined configurations for different scenarios:

- `RegionalObjectBindingConfig`: Object-attribute binding scenarios
- `AttributeBindingConfig`: Fine-grained attribute binding
- `ComplexSceneConfig`: Multi-object complex scenes
- `FastGenerationConfig`: Optimized for speed
- `HighQualityConfig`: Optimized for quality

## Usage Examples

### Basic Usage

```python
from configs.tome_flux_config import get_config
from tome_flux_pipeline import ToMeFluxPipeline

# Load configuration
config = get_config("regional_object_binding")

# Initialize pipeline
pipeline = ToMeFluxPipeline()
pipeline.configure_tome(
    tome_control_steps=config.tome_control_steps,
    token_refinement_steps=config.token_refinement_steps,
    scale_factor=config.scale_factor
)

# Generate images
result = pipeline(
    prompt=config.prompt,
    regional_prompts=config.regional_prompts,
    indices_to_alter=config.token_indices,
    height=config.height,
    width=config.width
)
```

### Running the Demo

```bash
# Basic demo with fast generation
python demo_tome_flux.py --config fast_generation

# High-quality generation
python demo_tome_flux.py --config high_quality --output-dir ./results

# Comparison demo
python demo_tome_flux.py --comparison

# Ablation study
python demo_tome_flux.py --ablation

# List available configurations
python demo_tome_flux.py --list-configs
```

## Configuration Options

### ToMe Control Parameters

- `tome_control_steps`: [token_steps, attention_steps] - Steps to apply ToMe refinement
- `token_refinement_steps`: Number of token refinement iterations per step
- `attention_refinement_steps`: [self_attn_steps, cross_attn_steps] - Attention refinement steps
- `eot_replace_step`: Step to replace end-of-text tokens
- `scale_factor`: Scale factor for gradient updates
- `merge_strategy`: Token merging strategy ("weighted_sum", "attention_weighted")

### Regional Control Parameters

- `regional_prompts`: List of prompts for different regions
- `token_indices`: Token indices to merge per region
- `prompt_merged`: Merged prompt for semantic consistency
- `cross_regional_binding`: Enable cross-regional semantic binding

### Generation Parameters

- `height`, `width`: Output image dimensions
- `num_inference_steps`: Number of denoising steps
- `guidance_scale`: Classifier-free guidance scale
- `seeds`: Random seeds for reproducible generation

## Advanced Features

### 1. Cross-Regional Binding

Maintains semantic consistency across different regions:

```python
# Enable cross-regional binding
config = get_config("complex_scene")
config.cross_regional_binding = True

# Automatic cross-regional token merging
enhanced_embeds = pipeline.apply_cross_regional_binding(regional_embeds)
```

### 2. Attention Map Analysis

Save and analyze attention maps for debugging:

```python
# Enable attention map saving
config.save_attention_maps = True

# Get attention maps after generation
attention_maps = pipeline.get_attention_maps(is_cross=True)
```

### 3. Dynamic EOT Replacement

Strategic end-of-text token replacement for better subject consistency:

```python
# Configure EOT replacement
pipeline.configure_tome(
    eot_replace_step=30,  # Replace after step 30
    # Earlier = risk of missing subjects
    # Later = risk of subject confusion
)
```

### 4. Iterative Refinement

Multi-step token and attention refinement:

```python
# Configure refinement steps
pipeline.configure_tome(
    token_refinement_steps=5,     # More iterations = better binding
    attention_refinement_steps=[8, 6],  # [self_attn, cross_attn]
)
```

## Performance Optimization

### Speed Optimization

```python
# Use FastGenerationConfig
config = get_config("fast_generation")
# - Reduced inference steps
# - Lower resolution
# - Minimal refinement iterations
```

### Quality Optimization

```python
# Use HighQualityConfig  
config = get_config("high_quality")
# - Extended inference steps
# - Higher resolution
# - More refinement iterations
```

### Memory Optimization

```python
# Reduce batch size and enable CPU offloading
pipeline.enable_model_cpu_offload()
config.num_images_per_prompt = 1
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size, image resolution, or enable CPU offloading
2. **Poor Binding**: Increase `token_refinement_steps` and `scale_factor`
3. **Subject Confusion**: Adjust `eot_replace_step` timing
4. **Regional Bleeding**: Refine regional masks and increase `attention_refinement_steps`

### Debug Mode

```python
# Enable attention map saving for debugging
config.save_attention_maps = True

# Check attention maps
attention_maps = pipeline.get_attention_maps()
print(f"Stored attention layers: {list(attention_maps.keys())}")
```

## Integration with Existing Code

### Extending Existing ToMe

```python
# Import existing ToMe utilities
from utils.ptp_utils import AttentionStore
from pipe_tome import tomePipeline

# Use FLUX adaptations
from tome_flux_utils import flux_token_merge
from tome_flux_pipeline import ToMeFluxPipeline

# Convert SDXL prompt setup to FLUX
sdxl_indices = [[[2], [3, 4]], [[7], [8, 9]]]  # From existing config
flux_embeds = flux_token_merge(prompt_embeds, sdxl_indices)
```

### Custom Configuration

```python
@dataclass 
class CustomToMeFluxConfig(ToMeFluxConfig):
    """Custom configuration for specific use case"""
    
    prompt: str = "your custom prompt"
    regional_prompts: List[str] = field(default_factory=lambda: [
        "custom region 1",
        "custom region 2"
    ])
    
    # Custom ToMe parameters
    tome_control_steps: List[int] = field(default_factory=lambda: [10, 8])
    scale_factor: float = 25.0
```

## Future Enhancements

- [ ] Support for dynamic regional mask generation
- [ ] Integration with ControlNet for precise regional control
- [ ] Automatic token index detection using NLP models
- [ ] Real-time attention visualization
- [ ] Batch processing for multiple prompts
- [ ] Integration with image editing workflows

## Citations

```bibtex
@article{hu2024token,
  title={Token Merging for Training-Free Semantic Binding in Text-to-Image Synthesis},
  author={Hu, Taihang and Li, Linxuan and van de Weijer, Joost and Gao, Hongcheng and Khan, Fahad and Yang, Jian and Cheng, Ming-Ming and Wang, Kai and Wang, Yaxing},
  journal={arXiv preprint arXiv:2411.07132},
  year={2024}
}
```

---

For more information about the original ToMe method, see the main [README.md](../README.md) file.