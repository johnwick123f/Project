from transformers import pipeline
import scipy
class musicgen:
    def __init__(self, musicgen_path="facebook/musicgen-small"):
        self.synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")
    
    def infer(self, prompt):
        music = self.synthesiser(prompt, forward_params={"do_sample": True})
        scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
