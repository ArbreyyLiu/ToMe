"""
Configuration classes for ToMe-FLUX integration.
Defines parameters for regional text-to-image generation with semantic binding.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass
class ToMeFluxConfig:
    """Base configuration for ToMe-FLUX pipeline"""
    
    # Model configuration
    model_path: str = "black-forest-labs/FLUX.1-dev"
    variant: str = "fp16"
    torch_dtype: str = "float16"
    
    # Generation parameters
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    
    # ToMe control parameters
    tome_control_steps: List[int] = field(default_factory=lambda: [5, 5])
    token_refinement_steps: int = 3
    attention_refinement_steps: List[int] = field(default_factory=lambda: [4, 4])
    eot_replace_step: int = 60
    
    # ToMe optimization parameters
    scale_factor: float = 20.0
    scale_range: Tuple[float, float] = field(default_factory=lambda: (1.0, 0.5))
    merge_strategy: str = "weighted_sum"
    temperature: float = 0.5
    
    # Regional control
    use_regional_control: bool = True
    cross_regional_binding: bool = True
    
    # Output configuration
    output_path: Path = Path("./demo_flux")
    save_attention_maps: bool = False
    seeds: List[int] = field(default_factory=lambda: [42, 123])
    
    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)


@dataclass
class RegionalObjectBindingConfig(ToMeFluxConfig):
    """Configuration for regional object binding scenarios"""
    
    # Main prompt
    prompt: str = "a red cat wearing blue sunglasses sitting next to a yellow dog wearing green hat"
    
    # Regional prompts for different areas
    regional_prompts: List[str] = field(default_factory=lambda: [
        "a red cat wearing blue sunglasses",
        "a yellow dog wearing green hat"
    ])
    
    # Token indices for merging [[object_tokens], [attribute_tokens]]
    token_indices: List[List[List[int]]] = field(default_factory=lambda: [
        [[[2], [3, 4]], [[7], [8, 9]]],  # Region 1: cat with sunglasses
        [[[2], [3, 4]], [[7], [8, 9]]]   # Region 2: dog with hat
    ])
    
    # Merged prompt for semantic consistency
    prompt_merged: str = "a cat and a dog"
    
    # Regional masks (to be generated or provided)
    mask_type: str = "auto"  # "auto", "manual", or "semantic"
    
    # ToMe-specific thresholds for this scenario
    thresholds: Dict[int, float] = field(default_factory=lambda: {
        0: 26, 1: 25, 2: 24, 3: 23, 4: 22.5,
        5: 22, 6: 21.5, 7: 21, 8: 21, 9: 21
    })
    
    # Enhanced binding for object-attribute association
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    eot_replace_step: int = 30
    use_pose_loss: bool = True


@dataclass
class AttributeBindingConfig(ToMeFluxConfig):
    """Configuration for fine-grained attribute binding"""
    
    # Main prompt focusing on attributes
    prompt: str = "a white fluffy cat with green eyes and a black sleek dog with brown eyes"
    
    # Regional prompts emphasizing different attributes
    regional_prompts: List[str] = field(default_factory=lambda: [
        "a white fluffy cat with green eyes",
        "a black sleek dog with brown eyes"
    ])
    
    # Fine-grained token indices for attribute binding
    token_indices: List[List[List[int]]] = field(default_factory=lambda: [
        [[[2], [3]], [[5], [6, 7]]],     # Region 1: white cat, green eyes
        [[[9], [10]], [[12], [13, 14]]]  # Region 2: black dog, brown eyes
    ])
    
    # Merged prompt
    prompt_merged: str = "a cat and a dog"
    
    # More conservative ToMe settings for attribute precision
    tome_control_steps: List[int] = field(default_factory=lambda: [5, 5])
    token_refinement_steps: int = 4
    attention_refinement_steps: List[int] = field(default_factory=lambda: [4, 4])
    eot_replace_step: int = 40
    
    # Lower scale factor for subtle adjustments
    scale_factor: float = 15.0
    temperature: float = 0.3
    
    # Emphasis on cross-regional consistency
    cross_regional_binding: bool = True
    use_pose_loss: bool = False


@dataclass
class ComplexSceneConfig(ToMeFluxConfig):
    """Configuration for complex multi-object scenes"""
    
    # Complex scene prompt
    prompt: str = "a red sports car parked next to a blue bicycle under a green tree with yellow flowers"
    
    # Multiple regional prompts
    regional_prompts: List[str] = field(default_factory=lambda: [
        "a red sports car",
        "a blue bicycle", 
        "a green tree with yellow flowers"
    ])
    
    # Multi-region token indices
    token_indices: List[List[List[int]]] = field(default_factory=lambda: [
        [[[2], [3, 4]]],     # Region 1: red sports car
        [[[6], [7]]],        # Region 2: blue bicycle
        [[[10], [11]], [[13], [14]]]  # Region 3: green tree, yellow flowers
    ])
    
    # Merged prompt for consistency
    prompt_merged: str = "a car, bicycle, and tree"
    
    # Extended control for complex scenes
    tome_control_steps: List[int] = field(default_factory=lambda: [10, 8])
    token_refinement_steps: int = 5
    attention_refinement_steps: List[int] = field(default_factory=lambda: [8, 6])
    eot_replace_step: int = 25
    
    # Higher scale factor for stronger control
    scale_factor: float = 25.0
    
    # More inference steps for complex scenes
    num_inference_steps: int = 75
    
    # Enable all advanced features
    cross_regional_binding: bool = True
    use_pose_loss: bool = True
    save_attention_maps: bool = True


@dataclass
class FastGenerationConfig(ToMeFluxConfig):
    """Configuration optimized for fast generation"""
    
    # Simple prompt for quick testing
    prompt: str = "a cute cat wearing sunglasses"
    
    # Single region
    regional_prompts: List[str] = field(default_factory=lambda: [
        "a cute cat wearing sunglasses"
    ])
    
    # Simple token merging
    token_indices: List[List[List[int]]] = field(default_factory=lambda: [
        [[[2], [3]], [[5], [6]]]  # cat with sunglasses
    ])
    
    # Minimal steps for speed
    num_inference_steps: int = 20
    tome_control_steps: List[int] = field(default_factory=lambda: [3, 3])
    token_refinement_steps: int = 1
    attention_refinement_steps: List[int] = field(default_factory=lambda: [2, 2])
    eot_replace_step: int = 15
    
    # Lower resolution for speed
    height: int = 512
    width: int = 512
    
    # Minimal refinement
    scale_factor: float = 10.0
    cross_regional_binding: bool = False
    use_pose_loss: bool = False


@dataclass 
class HighQualityConfig(ToMeFluxConfig):
    """Configuration for high-quality generation"""
    
    # Detailed prompt
    prompt: str = "a majestic golden retriever with silky fur wearing an elegant red velvet collar sitting in a sunlit garden"
    
    # Detailed regional prompts
    regional_prompts: List[str] = field(default_factory=lambda: [
        "a majestic golden retriever with silky fur",
        "an elegant red velvet collar",
        "a sunlit garden"
    ])
    
    # Detailed token indices
    token_indices: List[List[List[int]]] = field(default_factory=lambda: [
        [[[2, 3], [4]], [[6], [7, 8]]],  # golden retriever, silky fur
        [[[11], [12, 13, 14]]],          # elegant red velvet collar
        [[[17], [18, 19]]]               # sunlit garden
    ])
    
    # High-quality settings
    num_inference_steps: int = 100
    height: int = 1536
    width: int = 1536
    
    # Extended ToMe control
    tome_control_steps: List[int] = field(default_factory=lambda: [15, 12])
    token_refinement_steps: int = 8
    attention_refinement_steps: List[int] = field(default_factory=lambda: [10, 8])
    eot_replace_step: int = 20
    
    # Fine-tuned parameters
    scale_factor: float = 30.0
    temperature: float = 0.2
    
    # Enable all quality features
    cross_regional_binding: bool = True
    use_pose_loss: bool = True
    save_attention_maps: bool = True


# Configuration registry for easy access
TOME_FLUX_CONFIGS = {
    "regional_object_binding": RegionalObjectBindingConfig,
    "attribute_binding": AttributeBindingConfig,
    "complex_scene": ComplexSceneConfig,
    "fast_generation": FastGenerationConfig,
    "high_quality": HighQualityConfig
}


def get_config(config_name: str) -> ToMeFluxConfig:
    """Get configuration by name"""
    if config_name in TOME_FLUX_CONFIGS:
        return TOME_FLUX_CONFIGS[config_name]()
    else:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(TOME_FLUX_CONFIGS.keys())}")


def list_available_configs() -> List[str]:
    """List all available configuration names"""
    return list(TOME_FLUX_CONFIGS.keys())