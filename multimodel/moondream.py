# requires einops, transformers, timm
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
import torch
from utils import image_loader
def load_moondream(device="cuda:0"):
  model_id = "vikhyatk/moondream2"
  revision = "2024-03-06"
  moondream_model = AutoModelForCausalLM.from_pretrained(
      model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float16
  ).to(device)
  moondream_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
  return moondream_model, moondream_tokenizer
def moondream_inference(prompt="Describe the image.", image, caption=False):
  if caption == True
    image = image_loader(image, 'pil')
    enc_image = moondream_model.encode_image(image)
    out = moondream_model.answer_question(enc_image, "Describe this image.", moondream_tokenizer)
    return out
  else:
    image = image_loader(image, 'pil')
    enc_image = moondream_model.encode_image(image)
    out = moondream_model.answer_question(enc_image, prompt, moondream_tokenizer)
    return out
