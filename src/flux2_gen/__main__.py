import argparse
import random

from flux2_gen.pipeline import FluxGenerator

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate images using Flux2 with LoRA support.")
    
    parser.add_argument("--lora", type=str, help="The filename of the LoRA .safetensors file")
    

    parser.add_argument("--lora-weight", type=float, default=0.2, help="Strength of the LoRA adapter (default: 0.2)")

    parser.add_argument("--prompt", type=str, required=True, help="The text prompt for image generation")
    
    parser.add_argument("--output", type=str, default="flux2_output.png", help="The filename for the saved image")
    

    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Initialize Generator
    gen = FluxGenerator()

    # Load LoRA if provided
    if args.lora:
        gen.load_lora(args.lora, weight=args.lora_weight)


    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    # Generate
    image = gen.generate(
        prompt=args.prompt,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed
    )

    image.save(args.output)
    print(f"✅ Done! Saved to {args.output} (Seed: {args.seed})")

if __name__ == "__main__":
    main()