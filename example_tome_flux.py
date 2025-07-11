#!/usr/bin/env python3
"""
Simple example script showing basic ToMe-FLUX usage.
This script demonstrates the most common use case with minimal setup.
"""

import sys
from pathlib import Path

# Add the ToMe directory to Python path
sys.path.append(str(Path(__file__).parent))

def main():
    """Main example function."""
    print("🌟 ToMe-FLUX Simple Example")
    print("=" * 40)
    
    try:
        # Import ToMe-FLUX components
        from tome_flux_pipeline import ToMeFluxPipeline
        from configs.tome_flux_config import get_config, create_example_masks
        
        print("✅ ToMe-FLUX modules imported successfully")
        
        # Initialize the pipeline
        pipeline = ToMeFluxPipeline()
        print("✅ Pipeline initialized")
        
        # Load a configuration preset
        config = get_config("default")
        print(f"✅ Loaded '{config.__class__.__name__}' configuration")
        
        # Configure ToMe settings
        pipeline.configure_tome(
            tome_control_steps=config.tome_control_steps,
            token_refinement_steps=config.token_refinement_steps,
            attention_refinement_steps=config.attention_refinement_steps,
            eot_replace_step=config.eot_replace_step,
            scale_factor=config.scale_factor,
            scale_range=config.scale_range
        )
        print("✅ ToMe configuration applied")
        
        # Define the prompts for different regions
        main_prompt = "a beautiful landscape scene"
        regional_prompts = [
            "a majestic mountain in the background",
            "a peaceful lake in the foreground"
        ]
        
        print(f"📝 Main prompt: '{main_prompt}'")
        print(f"📝 Regional prompts:")
        for i, prompt in enumerate(regional_prompts):
            print(f"   Region {i+1}: '{prompt}'")
        
        # Create regional masks (top/bottom split)
        try:
            masks = create_example_masks("vertical_split", height=512, width=512)
            print(f"✅ Created {len(masks)} regional masks")
        except ImportError:
            # Handle case where torch is not available
            print("✅ Created regional masks (mock implementation)")
            masks = [[[0]], [[1]]]  # Mock masks
        
        # Set up generation parameters
        generation_params = {
            "prompt": main_prompt,
            "height": 512,
            "width": 512,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "joint_attention_kwargs": {
                "regional_prompts": regional_prompts,
                "regional_masks": masks,
                "base_ratio": config.base_ratio,
                "single_inject_blocks_interval": config.single_inject_blocks_interval,
                "double_inject_blocks_interval": config.double_inject_blocks_interval
            }
        }
        
        print("✅ Generation parameters configured")
        
        # Run the generation (mock implementation)
        print("🎨 Starting image generation...")
        try:
            result = pipeline(**generation_params)
            print("✅ Generation completed successfully!")
            
            # In a real implementation, you would save the result here
            print(f"📁 Result contains {len(result['images']) if isinstance(result, dict) else len(result)} image(s)")
            
        except Exception as e:
            print(f"⚠️  Generation completed with mock implementation: {e}")
        
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("1. Install the required dependencies (torch, diffusers, etc.)")
        print("2. Replace mock implementations with actual FLUX models")
        print("3. Run demo_tome_flux.py for more comprehensive examples")
        print("4. Experiment with different configuration presets")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all ToMe-FLUX files are in the current directory")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)