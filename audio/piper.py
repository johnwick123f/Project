## better alternative to musicgen. Only takes 2 seconds and slightly better then it? slightly more vram tho

from diffusers import AudioLDM2Pipeline
import torch
from diffusers import DPMSolverMultistepScheduler

class audio_generation:
    def __init__(self, model_path="cvssp/audioldm2"):  # Constructor
        self.pipe = AudioLDM2Pipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:0")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
    
    def infer(self, prompt, negative="low quality, average quality"):  # Another example method
        # Method logic here
        audio = self.pipe(prompt, num_inference_steps=20, audio_length_in_s=10.0, negative_prompt=negative).audios[0]
        #Audio(audio, rate=16000)
        return audio
