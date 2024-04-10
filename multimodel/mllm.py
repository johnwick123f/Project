from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
import torch
#from utils import image_loader


class moondream:
    def __init__(self, device="cuda:0"):
        self.moondream_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", trust_remote_code=True, revision="2024-04-0", torch_dtype=torch.float16
        ).to(device)
        self.moondream_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def infer(self, prompt, image, caption=True):
        if caption == True
            image = image_loader(image, 'pil')
            enc_image = moondream_model.encode_image(image)
            out = moondream_model.answer_question(enc_image, "Describe this image.", moondream_tokenizer)
            return out
        else:
            image = Image.open(image)
            enc_image = moondream_model.encode_image(image)
            out = moondream_model.answer_question(enc_image, prompt, moondream_tokenizer)
            return out
