## audio generation from text. fast and ezy
from piper import PiperVoice
import wave

class voice:
    def __init__(self, model='/content/en_US-lessac-high.onnx', config='/content/en_US-lessac-high.onnx.json'):
        self.voice = PiperVoice.load(model, config_path=config)
    
    def infer(text, file_out="test.wav"):
        with wave.open(file_out, "wb") as wav_file:
            self.voice.synthesize(text, wav_file)
