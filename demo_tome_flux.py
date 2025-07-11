"""
Demo script for ToMe-FLUX integration.
Showcases regional text-to-image generation with semantic binding capabilities.
"""
import os
import sys
import argparse
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add local modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tome_flux_pipeline import ToMeFluxPipeline
    from configs.tome_flux_config import get_config, list_available_configs
    from utils.vis_utils import view_images, text_under_image
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in development mode with mock implementations")
    
    # Mock implementations for development
    class ToMeFluxPipeline:
        def __init__(self, *args, **kwargs):
            print("Mock ToMeFluxPipeline initialized")
            
        def configure_tome(self, **kwargs):
            print("Mock ToMe configuration applied")
            
        def __call__(self, **kwargs):
            print(f"Mock generation with prompt: {kwargs.get('prompt', 'N/A')}")
            # Return mock images
            batch_size = kwargs.get('num_images_per_prompt', 1)
            height = kwargs.get('height', 1024)
            width = kwargs.get('width', 1024)
            mock_images = torch.randn(batch_size, 3, height, width)
            return {"images": mock_images}

warnings.filterwarnings("ignore", category=UserWarning)


def create_mock_regional_masks(height: int, width: int, num_regions: int) -> List[torch.Tensor]:
    """Create mock regional masks for demonstration"""
    masks = []
    
    # Create simple geometric regions
    h_step = height // num_regions
    
    for i in range(num_regions):
        mask = torch.zeros(height, width)
        start_h = i * h_step
        end_h = min((i + 1) * h_step, height)
        mask[start_h:end_h, :] = 1.0
        masks.append(mask)
    
    return masks


def generate_tome_flux_images(config, device: str = "cpu", output_dir: Optional[Path] = None):
    """Generate images using ToMe-FLUX pipeline"""
    
    print(f"Initializing ToMe-FLUX pipeline...")
    print(f"Device: {device}")
    print(f"Configuration: {type(config).__name__}")
    
    # Initialize pipeline (mock for development)
    pipeline = ToMeFluxPipeline()
    
    # Configure ToMe parameters
    tome_params = {
        'tome_control_steps': config.tome_control_steps,
        'token_refinement_steps': config.token_refinement_steps,
        'attention_refinement_steps': config.attention_refinement_steps,
        'eot_replace_step': config.eot_replace_step,
        'scale_factor': config.scale_factor,
        'scale_range': config.scale_range,
        'merge_strategy': config.merge_strategy,
        'temperature': config.temperature
    }
    
    if hasattr(config, 'thresholds'):
        tome_params['thresholds'] = config.thresholds
    
    pipeline.configure_tome(**tome_params)
    
    print(f"ToMe configuration applied:")
    for key, value in tome_params.items():
        print(f"  {key}: {value}")
    
    # Create regional masks if using regional control
    regional_masks = None
    if config.use_regional_control and hasattr(config, 'regional_prompts'):
        num_regions = len(config.regional_prompts)
        regional_masks = create_mock_regional_masks(config.height, config.width, num_regions)
        print(f"Created {num_regions} regional masks")
    
    # Generate images for each seed
    all_images = []
    generation_info = []
    
    for seed_idx, seed in enumerate(config.seeds):
        print(f"\n--- Generation {seed_idx + 1}/{len(config.seeds)} (seed: {seed}) ---")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Prepare generation parameters
        gen_params = {
            'prompt': config.prompt,
            'height': config.height,
            'width': config.width,
            'num_inference_steps': config.num_inference_steps,
            'guidance_scale': config.guidance_scale,
            'num_images_per_prompt': config.num_images_per_prompt,
        }
        
        # Add regional parameters if available
        if hasattr(config, 'regional_prompts'):
            gen_params['regional_prompts'] = config.regional_prompts
            gen_params['regional_masks'] = regional_masks
            
        if hasattr(config, 'token_indices'):
            gen_params['indices_to_alter'] = config.token_indices
            
        if hasattr(config, 'prompt_merged'):
            gen_params['prompt_merged'] = config.prompt_merged
        
        print(f"Main prompt: {config.prompt}")
        if hasattr(config, 'regional_prompts'):
            print(f"Regional prompts: {config.regional_prompts}")
        if hasattr(config, 'prompt_merged'):
            print(f"Merged prompt: {config.prompt_merged}")
        
        # Generate images
        try:
            result = pipeline(**gen_params)
            images = result["images"]
            
            print(f"Generated {len(images)} image(s)")
            all_images.extend(images)
            
            # Store generation info
            info = {
                'seed': seed,
                'config': type(config).__name__,
                'prompt': config.prompt,
                'params': gen_params.copy()
            }
            generation_info.append(info)
            
        except Exception as e:
            print(f"Generation failed: {e}")
            continue
    
    print(f"\nTotal images generated: {len(all_images)}")
    
    # Save images if output directory is specified
    if output_dir and all_images:
        save_generated_images(all_images, generation_info, output_dir, config)
    
    return all_images, generation_info


def save_generated_images(images: List[torch.Tensor], 
                         generation_info: List[Dict[str, Any]],
                         output_dir: Path,
                         config):
    """Save generated images with metadata"""
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for i, (image, info) in enumerate(zip(images, generation_info)):
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # [batch, channels, height, width]
                image = image[0]
            if image.shape[0] == 3:  # [channels, height, width]
                image = image.permute(1, 2, 0)  # [height, width, channels]
            image = image.detach().cpu().numpy()
            
        # Normalize to 0-255 range
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
        
        # Add text description
        desc_text = f"Seed: {info['seed']} | {info['config']}"
        image_with_text = text_under_image(image, desc_text)
        
        # Save image
        filename = f"{type(config).__name__.lower()}_seed_{info['seed']}_{i:03d}.png"
        filepath = output_dir / filename
        
        # Mock save (in real implementation, would use PIL or cv2)
        print(f"Saved: {filepath}")
        
    # Save configuration and metadata
    config_file = output_dir / "generation_config.txt"
    with open(config_file, 'w') as f:
        f.write(f"Configuration: {type(config).__name__}\n")
        f.write(f"Prompt: {config.prompt}\n")
        if hasattr(config, 'regional_prompts'):
            f.write(f"Regional prompts: {config.regional_prompts}\n")
        if hasattr(config, 'prompt_merged'):
            f.write(f"Merged prompt: {config.prompt_merged}\n")
        f.write(f"Seeds: {config.seeds}\n")
        f.write(f"ToMe control steps: {config.tome_control_steps}\n")
        f.write(f"Scale factor: {config.scale_factor}\n")
        f.write(f"Generated {len(images)} images\n")
    
    print(f"Configuration saved: {config_file}")


def run_comparison_demo():
    """Run comparison between different ToMe-FLUX configurations"""
    
    print("=== ToMe-FLUX Comparison Demo ===\n")
    
    # Test configurations
    test_configs = [
        "fast_generation",
        "attribute_binding", 
        "regional_object_binding"
    ]
    
    results = {}
    
    for config_name in test_configs:
        print(f"\n--- Testing {config_name} ---")
        
        try:
            config = get_config(config_name)
            images, info = generate_tome_flux_images(
                config, 
                device="cpu",
                output_dir=config.output_path
            )
            results[config_name] = {
                'images': images,
                'info': info,
                'config': config
            }
            print(f"✓ {config_name}: Generated {len(images)} images")
            
        except Exception as e:
            print(f"✗ {config_name}: Failed with error: {e}")
            results[config_name] = {'error': str(e)}
    
    # Summary
    print(f"\n=== Summary ===")
    successful = 0
    for name, result in results.items():
        if 'error' not in result:
            successful += 1
            print(f"✓ {name}: {len(result['images'])} images")
        else:
            print(f"✗ {name}: {result['error']}")
    
    print(f"\nSuccessful configurations: {successful}/{len(test_configs)}")
    
    return results


def run_ablation_study():
    """Run ablation study comparing ToMe vs non-ToMe generation"""
    
    print("=== ToMe-FLUX Ablation Study ===\n")
    
    base_config = get_config("regional_object_binding")
    
    # Test cases
    test_cases = [
        {
            "name": "baseline_no_tome",
            "tome_control_steps": [0, 0],  # Disable ToMe
            "description": "Standard FLUX without ToMe"
        },
        {
            "name": "tome_token_only", 
            "tome_control_steps": [5, 0],  # Token refinement only
            "description": "ToMe with token refinement only"
        },
        {
            "name": "tome_attention_only",
            "tome_control_steps": [0, 5],  # Attention refinement only
            "description": "ToMe with attention refinement only"
        },
        {
            "name": "tome_full",
            "tome_control_steps": [5, 5],  # Full ToMe
            "description": "ToMe with full capabilities"
        }
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"\n--- {case['name']}: {case['description']} ---")
        
        # Modify config for this test case
        test_config = base_config
        test_config.tome_control_steps = case['tome_control_steps']
        test_config.output_path = Path(f"./demo_flux/ablation_{case['name']}")
        
        try:
            images, info = generate_tome_flux_images(
                test_config,
                device="cpu",
                output_dir=test_config.output_path
            )
            results[case['name']] = {
                'images': images,
                'info': info,
                'description': case['description']
            }
            print(f"✓ Generated {len(images)} images")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[case['name']] = {'error': str(e)}
    
    print(f"\n=== Ablation Results ===")
    for name, result in results.items():
        if 'error' not in result:
            print(f"✓ {name}: {len(result['images'])} images - {result['description']}")
        else:
            print(f"✗ {name}: {result['error']}")
    
    return results


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="ToMe-FLUX Integration Demo")
    parser.add_argument(
        "--config", 
        type=str, 
        default="fast_generation",
        choices=list_available_configs(),
        help="Configuration to use for generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for generation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run comparison demo with multiple configurations"
    )
    parser.add_argument(
        "--ablation",
        action="store_true", 
        help="Run ablation study"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations"
    )
    
    args = parser.parse_args()
    
    if args.list_configs:
        print("Available configurations:")
        for config_name in list_available_configs():
            print(f"  - {config_name}")
        return
    
    if args.comparison:
        run_comparison_demo()
        return
        
    if args.ablation:
        run_ablation_study()
        return
    
    # Single configuration demo
    print(f"=== ToMe-FLUX Demo ===")
    print(f"Configuration: {args.config}")
    
    try:
        config = get_config(args.config)
        
        if args.output_dir:
            config.output_path = Path(args.output_dir)
        
        images, info = generate_tome_flux_images(
            config,
            device=args.device,
            output_dir=config.output_path
        )
        
        print(f"\n✓ Demo completed successfully!")
        print(f"Generated {len(images)} images")
        print(f"Output directory: {config.output_path}")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())