#!pip install einops
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
import torch
#from utils import image_loader


class moondream:
    def __init__(self, model_path="vikhyatk/moondream2", device="cuda:0"):
        self.moondream_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, revision="2024-04-02", torch_dtype=torch.float16
        ).to(device)
        self.moondream_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-04-02")

    def infer(self, prompt, image, caption=True):
        if caption == True:
            image = Image.open(image)
            enc_image = self.moondream_model.encode_image(image)
            out = self.moondream_model.answer_question(enc_image, "Describe this image.", self.moondream_tokenizer)
            return out
        else:
            image = Image.open(image)
            enc_image = self.moondream_model.encode_image(image)
            out = self.moondream_model.answer_question(enc_image, prompt, self.moondream_tokenizer)
            return out
