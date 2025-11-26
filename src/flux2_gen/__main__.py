import torch
import argparse
import sys

# NOTE: Flux2Pipeline and Flux2Transformer2DModel exist now
from diffusers import Flux2Pipeline, Flux2Transformer2DModel

from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io

def remote_text_encoder(prompts, device):
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        }
    )
    if response.status_code != 200:
        raise RuntimeError(f"Text Encoder API failed: {response.text}")
        
    prompt_embeds = torch.load(io.BytesIO(response.content), weights_only=False)
    return prompt_embeds.to(device)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate images using Flux2 with LoRA support.")
    
    parser.add_argument(
        "--lora", 
        type=str, 
        help="The filename of the LoRA .safetensors file"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        help="The text prompt for image generation"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="flux2_output.png",
        help="The filename for the saved image"
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    # 1. Configuration
    repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
    device = "cuda:0"
    torch_dtype = torch.bfloat16
    local_lora_dir = "." 
    
    # 2. Load Models
    print(f"Loading transformer from {repo_id}...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        torch_dtype=torch_dtype
    )

    print("Loading pipeline...")
    pipe = Flux2Pipeline.from_pretrained(
        repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype
    ).to(device)

    # 3. Load LoRA
    print(f"Loading LoRA: {args.lora}")
    try:
        pipe.load_lora_weights(local_lora_dir, weight_name=args.lora, adapter_name="my_target_style")
        pipe.set_adapters(["my_target_style"], adapter_weights=[0.2])
    except OSError:
        print(f"❌ Error: Could not find LoRA file '{args.lora}' in the current directory.")
        sys.exit(1)

    # 4. Generate
    print(f"Generating for prompt: '{args.prompt}'")
    
    # Pass device to helper function
    embeds = remote_text_encoder(args.prompt, device)
    
    image = pipe(
        prompt_embeds=embeds,
        generator=torch.Generator(device=device).manual_seed(42),
        num_inference_steps=50, 
        guidance_scale=4,
    ).images[0]

    # 5. Save
    image.save(args.output)
    print(f"✅ Done! Saved to {args.output}")

if __name__ == "__main__":
    main()
