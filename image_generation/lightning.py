import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderTiny
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class sdxl_lightning:
    def __init__(self, ckpt="sdxl_lightning_4step_unet.safetensors"):
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ## Very important loading this tiny vae since it saves roughly 4gb vram and slight increase in speed with almost no quality decrease.
        self.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl',use_safetensors=True,torch_dtype=torch.float16).to('cuda')
        self.unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
        self.unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
        self.pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=self.unet, torch_dtype=torch.float16, variant="fp16", vae=self.vae).to("cuda")
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

    
    def infer(self, prompt, save=True, steps=4, guidance_scale=0.75, negative_prompt="worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"):
        out = self.pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt)
        image = out.images[0]
        if save == True:
            image.save("output.png")
            return image
        else:
            return image
