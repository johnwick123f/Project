from styletts2 import tts
#Not much to change really, just slightly better usage.
class style_tts:
    def __init__(self, model_path="epochs_2nd_00020.pth", config_path='config.yml'):
        self.speech_model = tts.StyleTTS2(model_checkpoint_path=model_path, config_path=config_path)

    def infer(self, prompt, voice, out="another_test.wav"):
        self.speech_model.inference(prompt, target_voice_path=voice, output_wav_file=out)
