from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
from utils import image_loader
### Siglip is the best image classifier for its size. There are Various sizes 
class Siglip_model:
    def __init__(self, model_path="google/siglip-base-patch16-384", device="cuda:0"):
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device
  
    def infer(self, text, image):
        # Define behavior for method1
        img = image_loader(image, 'pil')
        inputs = processor(text=text, images=img, padding="max_length", return_tensors="pt").to(device, dtype=torch.float16)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image) # these are the probabilities
        return probs
        #print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
### Clip seg is a decent zero shot segmentation model. Quality is quite low but speed is extremely fast(0.1 seconds with float32)
class Clipseg_model:
    def __init__(self, model_path="CIDAS/clipseg-rd64-refined", device="cuda:0"):
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device
  
    def infer(self, text, image):
        # Define behavior for method1
        img = image_loader(image, 'pil')
        inputs = processor(text=text, images=[img] * len(text), padding="max_length", return_tensors="pt").to(self.device, dtype=torch.float16)
        outputs = model(**inputs)
        preds = outputs.logits.unsqueeze(1)
        return preds
