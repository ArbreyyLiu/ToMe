# ToMe-FLUX Integration: Regional Dream Renderer Enhancement

This implementation extends the Token Merging (ToMe) method from the SDXL pipeline to integrate with FLUX Regional Dream Renderer, enhancing semantic binding in regional text-to-image generation.

## 🚀 Features

- **Token Merging for FLUX**: Adapts the ToMe methodology for FLUX transformer architecture
- **Regional Attention Control**: Spatial awareness for different regions with separate prompts
- **Semantic Binding Enhancement**: Improved object-attribute associations through token fusion
- **Multiple Configuration Presets**: 6 different configurations optimized for various use cases
- **Pose Loss Support**: Spatial separation for multi-subject generation
- **Attention Refinement**: Entropy-based loss for improved attention focus

## 📁 File Structure

```
├── tome_flux_pipeline.py          # Main FLUX pipeline with ToMe capabilities
├── tome_flux_utils.py             # Token merging and semantic binding utilities
├── tome_flux_attention.py         # Regional attention processors
├── configs/tome_flux_config.py    # Configuration presets
├── demo_tome_flux.py              # Comprehensive demo script
└── test_tome_flux.py              # Validation tests
```

## 🛠️ Installation & Setup

1. **Install Dependencies** (when available):
   ```bash
   pip install torch torchvision diffusers transformers accelerate
   pip install numpy pillow
   ```

2. **Validate Installation**:
   ```bash
   python test_tome_flux.py
   ```

## 🎯 Usage

### Basic Usage

```python
from tome_flux_pipeline import ToMeFluxPipeline
from configs.tome_flux_config import get_config

# Initialize pipeline
pipe = ToMeFluxPipeline()

# Configure ToMe with preset
config = get_config("default")
pipe.configure_tome(
    tome_control_steps=config.tome_control_steps,
    token_refinement_steps=config.token_refinement_steps,
    attention_refinement_steps=config.attention_refinement_steps,
    scale_factor=config.scale_factor
)

# Define regional prompts and masks
regional_prompts = [
    "a red apple on the left",
    "a blue bird on the right"
]

# Create regional masks (example: horizontal split)
import torch
mask1 = torch.zeros(1024, 1024)
mask1[:, :512] = 1.0  # Left half

mask2 = torch.zeros(1024, 1024) 
mask2[:, 512:] = 1.0  # Right half

regional_masks = [mask1, mask2]

# Generate image
output = pipe(
    prompt="a nature scene",
    height=1024,
    width=1024,
    num_inference_steps=25,
    joint_attention_kwargs={
        "regional_prompts": regional_prompts,
        "regional_masks": regional_masks,
        "base_ratio": 0.3
    }
)
```

### Advanced Configuration

```python
# Use high-quality preset for best results
config = get_config("high_quality")

# Or customize parameters
config = get_config("default", 
                   scale_factor=30.0,
                   num_inference_steps=40,
                   use_pose_loss=True)
```

## ⚙️ Configuration Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `default` | Balanced settings | General purpose |
| `high_binding` | Strong semantic binding | Complex prompts with many attributes |
| `fast` | Speed optimized | Quick generation |
| `high_quality` | Maximum quality | Best results, slower |
| `regional_focus` | Enhanced regional control | Spatially-aware generation |
| `multi_subject` | Multiple subject separation | Clear subject boundaries |

## 🔧 Key Parameters

### ToMe Control Parameters
- `tome_control_steps`: [token_steps, attention_steps] - When to apply ToMe processing
- `token_refinement_steps`: Number of token optimization iterations
- `attention_refinement_steps`: Attention map refinement iterations
- `eot_replace_step`: When to replace end-of-text tokens
- `scale_factor`: Strength of latent updates
- `scale_range`: [start, end] - Decay range for scale factor

### Regional Control Parameters
- `base_ratio`: Balance between base and regional prompts
- `single_inject_blocks_interval`: Interval for single-block injection
- `double_inject_blocks_interval`: Interval for double-block injection
- `use_pose_loss`: Enable spatial separation loss

## 🎨 Mask Creation

### Built-in Mask Types

```python
from configs.tome_flux_config import create_example_masks

# Horizontal split
masks = create_example_masks("horizontal_split", 1024, 1024)

# Vertical split  
masks = create_example_masks("vertical_split", 1024, 1024)

# Center with surround
masks = create_example_masks("center_surround", 1024, 1024)

# Four quadrants
masks = create_example_masks("quadrants", 1024, 1024)
```

### Custom Masks

```python
import torch
import numpy as np

# Create custom circular mask
height, width = 1024, 1024
center_y, center_x = height // 2, width // 2
radius = 200

y, x = np.ogrid[:height, :width]
mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
mask_tensor = torch.from_numpy(mask.astype(np.float32))
```

## 🧪 Demo Script

Run the comprehensive demo to see all features:

```bash
python demo_tome_flux.py
```

The demo includes:
- Basic ToMe-FLUX generation
- Multiple configuration comparisons
- Different regional mask patterns
- Token merging strategy analysis

## 🔍 Technical Implementation

### Token Merging Process

1. **Token Identification**: Parse prompt to identify object-attribute relationships
2. **Semantic Fusion**: Merge related tokens using weighted combination
3. **Attention Refinement**: Optimize attention maps for better focus
4. **Regional Application**: Apply merging within specific spatial regions

### Attention Processing

1. **Regional Attention**: Apply different prompts to different spatial areas
2. **Cross-Attention Enhancement**: Improve object-attribute binding
3. **Entropy Minimization**: Encourage focused attention distributions
4. **Pose Loss**: Maintain spatial separation between subjects

### FLUX Adaptations

- Modified token merging for FLUX embedding dimensions
- Adapted attention processors for FLUX transformer blocks
- Regional mask handling for FLUX spatial representations
- Custom loss functions for FLUX training dynamics

## 📊 Validation & Testing

The implementation includes comprehensive validation:

```bash
python test_tome_flux.py
```

Tests cover:
- ✅ Module imports and dependencies
- ✅ Configuration loading and validation  
- ✅ Utility function correctness
- ✅ Pipeline initialization and setup
- ✅ Mock implementation for missing dependencies

## 🎛️ Configuration Validation

```python
from configs.tome_flux_config import validate_config

config = get_config("default")
warnings = validate_config(config)
if warnings:
    print("Configuration warnings:", warnings)
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: The implementation includes mock objects for missing dependencies
2. **Memory Issues**: Use `fast` preset for lower memory usage
3. **Quality Issues**: Try `high_quality` or `high_binding` presets
4. **Regional Blending**: Adjust `base_ratio` and injection intervals

### Performance Tips

- Use `fast` preset for quick iterations
- Reduce `num_inference_steps` for speed
- Lower `attention_res` for memory efficiency
- Disable `save_attention_maps` in production

## 🤝 Integration with Existing Code

The ToMe-FLUX implementation is designed to be compatible with the existing ToMe codebase:

```python
# Existing ToMe parameters can be used
from configs.demo_config import RunConfig1

# Convert to ToMe-FLUX config
flux_config = get_config("default",
                        tome_control_steps=RunConfig1.tome_control_steps,
                        token_refinement_steps=RunConfig1.token_refinement_steps,
                        scale_factor=RunConfig1.scale_factor)
```

## 📈 Future Enhancements

- [ ] Integration with actual FLUX models when available
- [ ] Real-time attention visualization
- [ ] Automatic mask generation from prompts
- [ ] Performance optimizations for large-scale generation
- [ ] Advanced loss functions for better semantic binding

## 📄 License

This implementation follows the same license as the original ToMe repository.

## 🙏 Acknowledgments

Built upon the excellent work from:
- Original ToMe implementation for SDXL
- FLUX transformer architecture
- Regional Dream Renderer concepts
- Diffusers library ecosystem

---

For more details, see the individual module documentation in the source files.