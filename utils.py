from PIL import Image
import numpy as np
import os
import requests
from sentence_transformers import SentenceTransformer, util
import torch
import re
## A class for using a sentence embedder model to find similar sentences(can be used for classification, clustering and more) 
class Similarity:
    # Class variable to store the loaded model
    embedder = None

    def __init__(self, model_path="all-MiniLM-L6-v2", device="cuda:0"):
        self.device = device
        # Load the model if not already loaded
        if Similarity.embedder is None:
            Similarity.embedder = SentenceTransformer(model_path).to(device)

    def infer(self, queries, corpus, top=1):
        corpus_embeddings = Similarity.embedder.encode(corpus, convert_to_tensor=True).to(self.device)
        top_k = min(top, len(corpus))
        results=[]
        for query in queries:
            query_embedding = Similarity.embedder.encode(query, convert_to_tensor=True).to(self.device)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            for score, idx in zip(top_results[0], top_results[1]):
                results.append(corpus[idx])
        return results
# loads any sort of image but kinda useless                
def image_loader(image, to_format):
    # Load the image
    if isinstance(image, str):
        if str.startswith("http"):
            img = Image.open(requests.get(image, stream=True).raw)
        else:
            img = Image.open(image)
    elif isinstance(image, np.ndarray):  # If image is a NumPy array
        img = Image.fromarray(image)
    elif isinstance(image, Image.Image):  # If image is a PIL Image object
        img = image
    else:
        raise ValueError("Unsupported image type. It should be a file path, NumPy array, or PIL Image object.")

    # Convert to the specified format
    if to_format == 'path':
        # Save image to a temporary file
        temp_path = 'temp_image.jpg'
        img.save(temp_path)
        return temp_path
    elif to_format == 'np':
        return np.array(img)
    elif to_format == 'pil':
        return img
    else:
        raise ValueError("Invalid format. It should be one of 'path', 'np', or 'pil'.")
