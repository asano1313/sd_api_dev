from fastapi import FastAPI
from pydantic import BaseModel

from diffusers import DiffusionPipeline
import torch
import os
import base64
import signal

app = FastAPI(docs_url="/docs")

class Text2ImgPrm(BaseModel):
    prompt: str
    seed: int

@app.get('/')
def index():
    # pipe = DiffusionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    #     use_safetensors=True
    # )
    # pipe.to("cuda")
    return {"message": "OKOK"}



@app.post('/text2img')
async def text2img(_text2imgprm: Text2ImgPrm):
    prompt = _text2imgprm.prompt
    seed = _text2imgprm.seed
    
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        # "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16")
    pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    image = pipe(
    prompt=prompt,
    generator=generator).images[0]

    image.save("result_0.png")
    with open('result_0.png', 'rb') as f:
        base64_img = base64.b64encode(f.read()).decode('utf-8')
    os.remove('result_0.png')
    # image = pipe(
    #     prompt=prompt,
    #     generator=generator
    # )

    # print(image)
    del pipe, generator, image
    torch.cuda.empty_cache()

    pid = os.getpid()
    os.system(f"nvidia-smi --gpu-reset -i 0 -r -c {pid}")
    os.kill(pid, signal.SIGTERM)

    return base64_img