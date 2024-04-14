import torch, torchvision
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers import AutoencoderTiny
import os
#%cd piecewise-rectified-flow
from src.scheduler_perflow import PeRFlowScheduler
class SDFast:
    def __init__(self, model_path="hansyan/perflow-sd15-dreamshaper"):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, device="cuda:0", safety_checker=None)
        self.pipe.scheduler = PeRFlowScheduler.from_config(self.pipe.scheduler.config, prediction_type="epsilon", num_time_windows=4)
        self.pipe.safety_checker = None
        self.pipe.to(torch.float16, device="cuda:0")

    def infer(self, prompt, num_images=1, mode="portrait", guidance=7.5, num_steps=8, neg="worst quality"):
        #prompt = [f"RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, (masterpiece), cinematic, vivid colors, intricate masterpiece, golden ratio, highly detailed; {prompt}"]
        prompt = [f"{prompt}((8k, RAW photo, highest quality, masterpiece), High detail RAW color photo professional close-up photo, (realistic, photo realism:1.4), (highest quality), (best shadow), (best illustration), ultra high resolution, highly detailed CG unified 8K wallpapers, physics-based rendering, cinematic lighting)"]
        neg_prompt = [f"lowres, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, "]
        if mode == "portrait":
            height=512
            width=512
        else:
            height=512
            width=1024
        samples = self.pipe(
            prompt              = prompt, 
            negative_prompt     = neg_prompt,
            height              = height,
            width               = width,
            num_inference_steps = num_steps, 
            guidance_scale      = guidance,
            num_images_per_prompt=num_images
         ).images
        image_grid(samples, rows=1, cols=num_images)
        return samples
#sdfast = SDFast()
