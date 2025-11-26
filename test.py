import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

def remote_text_encoder(prompts):
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        }
    )
    prompt_embeds = torch.load(io.BytesIO(response.content))

    return prompt_embeds.to(device)

pipe = Flux2Pipeline.from_pretrained(
    repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype
).to(device)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    image=load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png"),
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50, #28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
