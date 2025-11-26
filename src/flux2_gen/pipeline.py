import torch
import requests
import io
import sys
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from diffusers.utils import load_image
from huggingface_hub import get_token

class FluxGenerator:
    def __init__(self, model_id="diffusers/FLUX.2-dev-bnb-4bit", device="cuda:0"):
        self.device = device
        self.dtype = torch.bfloat16

        print(f"Loading transformer from {model_id}...")

        self.transformer = Flux2Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=self.dtype
        )

        print("Loading pipeline...")
        self.pipe = Flux2Pipeline.from_pretrained(
            model_id, 
            transformer=self.transformer,
            text_encoder=None, 
            torch_dtype=self.dtype
        ).to(self.device)

    def load_lora(self, lora_path, weight=0.2, adapter_name="target_style"):
        print(f"Loading LoRA: {lora_path} with weight {weight}")
        try:
            self.pipe.load_lora_weights(".", weight_name=lora_path, adapter_name=adapter_name)
            self.pipe.set_adapters([adapter_name], adapter_weights=[weight])
        except Exception as e:
            print(f"❌ Error loading LoRA: {e}")
            sys.exit(1)

    def _remote_text_encoder(self, prompt):
        response = requests.post(
            "https://remote-text-encoder-flux-2.huggingface.co/predict",
            json={"prompt": prompt},
            headers={
                "Authorization": f"Bearer {get_token()}",
                "Content-Type": "application/json"
            },
            timeout=30 
        )
        if response.status_code != 200:
            raise RuntimeError(f"Text Encoder API failed: {response.text}")
            
        return torch.load(io.BytesIO(response.content), weights_only=False).to(self.device)

    def generate(self, prompt, image_path=None, steps=50, guidance=4.0, seed=None):
        print(f"Generating: '{prompt}'")
        
        embeds = self._remote_text_encoder(prompt)
        
        if seed is None:
            generator = None
        else:
            print(f"Using seed: {seed}")
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Prepare arguments common to both modes
        generate_kwargs = {
            "prompt_embeds": embeds,
            "generator": generator,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }

        if image_path:
            print(f"🖼️ Running Img2Img on {image_path}")
            init_image = load_image(image_path)
            generate_kwargs["image"] = init_image

        return self.pipe(**generate_kwargs).images[0]
