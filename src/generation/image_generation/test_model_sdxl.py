from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
import time

# 개선된 VAE 로드 (흐릿함 해결)
print("Loading improved VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

print("Loading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,  # 개선된 VAE 사용
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

prompt = """
professional coffee shop advertisement poster,
cozy modern cafe interior with warm wooden furniture,
ceramic coffee cups and latte art on rustic wooden table,
chalkboard menu with elegant typography on brick wall,
soft golden hour natural lighting through window,
warm brown and cream color palette, inviting atmosphere,
commercial product photography, magazine quality,
sharp focus, highly detailed, 4k resolution
"""

negative_prompt = """
low quality, blurry, distorted, ugly, deformed, bad anatomy,
watermark, text overlay, signature, logo, amateur photo,
low resolution, oversaturated colors, cartoon, anime style,
3d render, plastic looking, artificial, people, faces
"""

print("Generating image...")
start_time = time.time()

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=7.5,
    height=1024,
    width=1024,
).images[0]

elapsed_time = time.time() - start_time

image.save('test_output.png')
print(f"\n✅ Success! Generated in {elapsed_time:.2f} seconds")
print(f"📐 Image size: {image.size}")
print(f"💾 Saved to: test_output.png")
