# PerFlow Stable Diffusion
Description: A model to generate images
Vram:
- Without TinyVAE: 3.4gb vram
- With TinyVAE(not reccomended): 3 gb vram
Speed:
- 1 second for 512x512 1 images
- 3 seconds for 512x512 4 images
- 2 seconds for 512x1024 1 image
Timing
# Moondream
Description: a model to caption and vqa images
Vram:
- takes roughly 3gb vram
Speed:
- caption takes a second
- vqa is fast/faster then blip
# Distilled whisper v3
Description: converts speech to text
Vram:
- 1.8gb vram
Speed:
- 0.3 seconds for 2 words so easily fast enough.
# Hermes Pro 7b(language model like chatgpt but 20x smaller)
Description: answers question, generates story and more
Vram:
- 4.9gb vram
Speed:
- 40 tokens per second(0.75 tokens = 1 word)
