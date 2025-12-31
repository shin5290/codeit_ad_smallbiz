from diffusers import FluxPipeline
import torch
import time

# Hugging Face 로그인 필요
# 터미널에서 먼저 실행: huggingface-cli login
# 또는 프로젝트 루트에 .env 파일 생성 후 HF_TOKEN=your_token 추가

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
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
    num_inference_steps=28,
    guidance_scale=7.5,
    height=1024,
    width=1024,
).images[0]

elapsed_time = time.time() - start_time

image.save('test_output_flux.png')
print(f"\n✅ Success! Generated in {elapsed_time:.2f} seconds")
print(f"📐 Image size: {image.size}")
print(f"💾 Saved to: test_output_flux.png")
