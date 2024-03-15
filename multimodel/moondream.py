# requires einops, transformers, timm
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
import torch
def load_moondream():
  model_id = "vikhyatk/moondream2"
  revision = "2024-03-06"
  model = AutoModelForCausalLM.from_pretrained(
      model_id, trust_remote_code=True, revision=revision
  )
  tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
  return model, tokenizer
