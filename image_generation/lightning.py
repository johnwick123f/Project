import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class sdxl_lightning:
    def __init__(self, ckpt="sdxl_lightning_4step_unet.safetensors"):
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        # Load model.
        self.unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
        self.unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
        self.pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

    
    def infer(self, prompt, save=True, steps=4, guidance_scale=0):
        out = self.pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale)
        image = out.images[0]
        if save == True:
            image.save("output.png")
            return image
        else:
            return image
