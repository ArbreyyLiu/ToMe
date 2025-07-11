"""
Configuration presets for ToMe-FLUX integration with different use cases.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    # Mock torch for environments where it's not available
    class MockTensor:
        def __init__(self, data):
            self.data = data
    
    torch = type('MockTorch', (), {
        'zeros': lambda *args, **kwargs: MockTensor([[0]]),
        'from_numpy': lambda x: MockTensor(x),
        'stack': lambda x, **kwargs: MockTensor(x),
        'Tensor': MockTensor,
    })()
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Mock numpy
    np = type('MockNumpy', (), {
        'ogrid': [[[0]], [[0]]],
        'sqrt': lambda x: x,
        'float32': float,
    })()
    NUMPY_AVAILABLE = False


@dataclass
class ToMeFluxConfig:
    """Base configuration class for ToMe-FLUX integration."""
    
    # Model and generation settings
    model_path: str = "black-forest-labs/FLUX.1-schnell"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 25
    guidance_scale: float = 3.5
    
    # ToMe-specific parameters
    tome_control_steps: List[int] = field(default_factory=lambda: [5, 5])
    token_refinement_steps: int = 3
    attention_refinement_steps: List[int] = field(default_factory=lambda: [4, 4])
    eot_replace_step: int = 60
    thresholds: Optional[List[float]] = None
    scale_factor: float = 20.0
    scale_range: List[float] = field(default_factory=lambda: [1.0, 0.5])
    
    # Regional control parameters
    base_ratio: float = 0.3
    single_inject_blocks_interval: int = 2
    double_inject_blocks_interval: int = 2
    attention_res: int = 32
    
    # Advanced settings
    use_pose_loss: bool = False
    smooth_attentions: bool = True
    sigma: float = 0.5
    kernel_size: int = 3
    run_standard_sd: bool = False
    
    # Output settings
    output_path: Path = Path("./outputs/tome_flux")
    save_attention_maps: bool = False
    save_intermediate_results: bool = False
    
    def __post_init__(self):
        """Initialize computed fields and validate configuration."""
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        if self.thresholds is None:
            # Default adaptive thresholds
            self.thresholds = [8.0 - 0.1 * i for i in range(self.num_inference_steps)]
        
        # Ensure thresholds list matches inference steps
        if len(self.thresholds) < self.num_inference_steps:
            # Extend with last value
            last_threshold = self.thresholds[-1] if self.thresholds else 5.0
            self.thresholds.extend([last_threshold] * (self.num_inference_steps - len(self.thresholds)))


@dataclass 
class DefaultToMeFluxConfig(ToMeFluxConfig):
    """Default balanced configuration for general use cases."""
    
    # Balanced ToMe parameters
    tome_control_steps: List[int] = field(default_factory=lambda: [5, 5])
    token_refinement_steps: int = 3
    attention_refinement_steps: List[int] = field(default_factory=lambda: [4, 4])
    eot_replace_step: int = 60
    scale_factor: float = 20.0
    scale_range: List[float] = field(default_factory=lambda: [1.0, 0.5])
    
    # Standard regional parameters
    base_ratio: float = 0.3
    use_pose_loss: bool = False


@dataclass
class HighBindingToMeFluxConfig(ToMeFluxConfig):
    """Configuration optimized for strong semantic binding."""
    
    # Aggressive ToMe parameters for stronger binding
    tome_control_steps: List[int] = field(default_factory=lambda: [10, 10])
    token_refinement_steps: int = 5
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    eot_replace_step: int = 40
    scale_factor: float = 25.0
    scale_range: List[float] = field(default_factory=lambda: [1.2, 0.6])
    
    # Enhanced regional control
    base_ratio: float = 0.4
    use_pose_loss: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        # More aggressive thresholds for stronger binding
        self.thresholds = [6.0 - 0.05 * i for i in range(self.num_inference_steps)]


@dataclass
class FastToMeFluxConfig(ToMeFluxConfig):
    """Configuration optimized for faster generation with moderate quality."""
    
    # Reduced inference steps for speed
    num_inference_steps: int = 15
    guidance_scale: float = 3.0
    
    # Lighter ToMe processing
    tome_control_steps: List[int] = field(default_factory=lambda: [3, 3])
    token_refinement_steps: int = 2
    attention_refinement_steps: List[int] = field(default_factory=lambda: [2, 2])
    eot_replace_step: int = 50
    scale_factor: float = 15.0
    scale_range: List[float] = field(default_factory=lambda: [0.8, 0.3])
    
    # Simplified regional control
    base_ratio: float = 0.2
    use_pose_loss: bool = False
    smooth_attentions: bool = False


@dataclass
class HighQualityToMeFluxConfig(ToMeFluxConfig):
    """Configuration optimized for highest quality output."""
    
    # Extended inference for quality
    num_inference_steps: int = 40
    guidance_scale: float = 4.0
    
    # Intensive ToMe processing
    tome_control_steps: List[int] = field(default_factory=lambda: [15, 15])
    token_refinement_steps: int = 6
    attention_refinement_steps: List[int] = field(default_factory=lambda: [8, 8])
    eot_replace_step: int = 30
    scale_factor: float = 30.0
    scale_range: List[float] = field(default_factory=lambda: [1.5, 0.8])
    
    # Fine-grained regional control
    base_ratio: float = 0.5
    single_inject_blocks_interval: int = 1
    double_inject_blocks_interval: int = 1
    use_pose_loss: bool = True
    smooth_attentions: bool = True
    attention_res: int = 64
    
    # Enhanced analysis
    save_attention_maps: bool = True
    save_intermediate_results: bool = True


@dataclass
class RegionalFocusConfig(ToMeFluxConfig):
    """Configuration optimized for strong regional control."""
    
    # Standard ToMe parameters
    tome_control_steps: List[int] = field(default_factory=lambda: [6, 6])
    token_refinement_steps: int = 4
    attention_refinement_steps: List[int] = field(default_factory=lambda: [5, 5])
    
    # Strong regional emphasis
    base_ratio: float = 0.6
    single_inject_blocks_interval: int = 1
    double_inject_blocks_interval: int = 1
    use_pose_loss: bool = True
    
    # Higher resolution for better regional control
    attention_res: int = 64
    smooth_attentions: bool = True
    sigma: float = 0.3
    kernel_size: int = 5


@dataclass
class MultiSubjectConfig(ToMeFluxConfig):
    """Configuration optimized for multiple subjects with clear separation."""
    
    # Enhanced separation parameters
    tome_control_steps: List[int] = field(default_factory=lambda: [8, 8])
    token_refinement_steps: int = 4
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    eot_replace_step: int = 20  # Earlier EOT replacement for better subject definition
    
    # Strong pose loss for subject separation
    use_pose_loss: bool = True
    scale_factor: float = 25.0
    scale_range: List[float] = field(default_factory=lambda: [1.3, 0.7])
    
    # Balanced regional control
    base_ratio: float = 0.4
    single_inject_blocks_interval: int = 2
    double_inject_blocks_interval: int = 2


# Dictionary of all available configurations
TOME_FLUX_CONFIGS = {
    "default": DefaultToMeFluxConfig,
    "high_binding": HighBindingToMeFluxConfig,
    "fast": FastToMeFluxConfig,
    "high_quality": HighQualityToMeFluxConfig,
    "regional_focus": RegionalFocusConfig,
    "multi_subject": MultiSubjectConfig,
}


def get_config(config_name: str, **kwargs) -> ToMeFluxConfig:
    """
    Get a configuration instance by name with optional parameter overrides.
    
    Args:
        config_name: Name of the configuration preset
        **kwargs: Parameter overrides
        
    Returns:
        Configuration instance
        
    Raises:
        ValueError: If config_name is not found
    """
    if config_name not in TOME_FLUX_CONFIGS:
        available_configs = list(TOME_FLUX_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available_configs}")
    
    config_class = TOME_FLUX_CONFIGS[config_name]
    config = config_class(**kwargs)
    
    return config


def list_available_configs() -> List[str]:
    """Return list of available configuration names."""
    return list(TOME_FLUX_CONFIGS.keys())


def get_config_description(config_name: str) -> str:
    """
    Get a human-readable description of a configuration.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Description string
    """
    descriptions = {
        "default": "Balanced configuration for general use cases with moderate semantic binding",
        "high_binding": "Strong semantic binding with aggressive token merging for complex prompts",
        "fast": "Optimized for speed with reduced quality but faster generation",
        "high_quality": "Maximum quality with intensive processing and fine-grained control",
        "regional_focus": "Emphasizes regional control for spatially-aware generation",
        "multi_subject": "Optimized for multiple subjects with clear spatial separation",
    }
    
    return descriptions.get(config_name, "No description available")


# Example configurations for specific use cases
EXAMPLE_REGIONAL_CONFIGS = {
    "two_subjects_horizontal": {
        "regional_prompts": [
            "a red apple on the left side",
            "a blue bird on the right side"
        ],
        "mask_type": "horizontal_split",
        "base_ratio": 0.3,
    },
    
    "foreground_background": {
        "regional_prompts": [
            "a detailed portrait in the foreground",
            "a beautiful landscape in the background"
        ],
        "mask_type": "depth_based",
        "base_ratio": 0.4,
    },
    
    "center_surround": {
        "regional_prompts": [
            "a castle in the center",
            "a magical forest surrounding"
        ],
        "mask_type": "center_surround",
        "base_ratio": 0.5,
    },
}


def create_example_masks(mask_type: str, height: int = 1024, width: int = 1024) -> List:
    """
    Create example masks for different regional configurations.
    
    Args:
        mask_type: Type of mask to create
        height: Mask height
        width: Mask width
        
    Returns:
        List of mask tensors
    """
    import torch
    import numpy as np
    import numpy as np
    
    if mask_type == "horizontal_split":
        # Split horizontally into left and right halves
        mask1 = torch.zeros(height, width)
        mask1[:, :width//2] = 1.0
        
        mask2 = torch.zeros(height, width)
        mask2[:, width//2:] = 1.0
        
        return [mask1, mask2]
    
    elif mask_type == "vertical_split":
        # Split vertically into top and bottom halves
        mask1 = torch.zeros(height, width)
        mask1[:height//2, :] = 1.0
        
        mask2 = torch.zeros(height, width)
        mask2[height//2:, :] = 1.0
        
        return [mask1, mask2]
    
    elif mask_type == "center_surround":
        # Center circle and surrounding area
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 4
        
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        mask1 = torch.from_numpy((distance <= radius).astype(np.float32))
        mask2 = torch.from_numpy((distance > radius).astype(np.float32))
        
        return [mask1, mask2]
    
    elif mask_type == "quadrants":
        # Four quadrants
        masks = []
        for i in range(2):
            for j in range(2):
                mask = torch.zeros(height, width)
                start_y = i * height // 2
                end_y = (i + 1) * height // 2
                start_x = j * width // 2
                end_x = (j + 1) * width // 2
                mask[start_y:end_y, start_x:end_x] = 1.0
                masks.append(mask)
        
        return masks
    
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


# Utility function to validate configuration
def validate_config(config: ToMeFluxConfig) -> List[str]:
    """
    Validate a configuration and return list of warnings/errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation messages
    """
    warnings = []
    
    # Check tome_control_steps
    if len(config.tome_control_steps) != 2:
        warnings.append("tome_control_steps should have exactly 2 elements [token_steps, attention_steps]")
    
    # Check scale_range
    if len(config.scale_range) != 2:
        warnings.append("scale_range should have exactly 2 elements [start_scale, end_scale]")
    elif config.scale_range[0] < config.scale_range[1]:
        warnings.append("scale_range start value should be >= end value for proper decay")
    
    # Check thresholds length
    if config.thresholds and len(config.thresholds) != config.num_inference_steps:
        warnings.append(f"thresholds length ({len(config.thresholds)}) doesn't match num_inference_steps ({config.num_inference_steps})")
    
    # Check EOT replacement step
    if config.eot_replace_step >= config.num_inference_steps:
        warnings.append("eot_replace_step should be less than num_inference_steps")
    
    # Check refinement steps
    if config.token_refinement_steps <= 0:
        warnings.append("token_refinement_steps should be positive")
    
    if len(config.attention_refinement_steps) != 2:
        warnings.append("attention_refinement_steps should have exactly 2 elements")
    
    return warnings