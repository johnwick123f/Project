import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
class whisperHF:
    def __init__(self, model_path='distil-whisper/distil-large-v3', device="cuda:0"):
        self.device = device
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
        ).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.pipe = pipeline(
        "automatic-speech-recognition",
        model=self.model,
        tokenizer=self.processor.tokenizer,
        feature_extractor=self.processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch.float16,
        device=self.device,
        )
    
    def infer(self, file="sound.mp3"):
        result = self.pipe(file)
        return result["text"]
    
    def infer_timestep(self, file="sound.mp3"):
        result = pipe(sample, return_timestamps=True)
        return result["chunks"]
