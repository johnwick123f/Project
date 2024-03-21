from PIL import Image
import numpy as np
import os
import requests
def image_loader(image, to_format):
    """
    Load an image from path, NumPy array, or PIL Image object and convert it to the specified format.

    Parameters:
        image: str, np.ndarray, PIL.Image.Image
            The input image, which can be a file path, NumPy array, or PIL Image object.
        to_format: str
            The format to convert the image to. Can be one of 'path', 'np', or 'pil'.

    Returns:
        Converted image in the specified format.
    """
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
