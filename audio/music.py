from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np
import torch

from transformers import MusicgenForConditionalGeneration, MusicgenProcessor, set_seed
from transformers.generation.streamers import BaseStreamer
class MusicgenStreamer(BaseStreamer):
    def __init__(
        self,
        model: MusicgenForConditionalGeneration,
        device: Optional[str] = None,
        play_steps: Optional[int] = 10,
        stride: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Streamer that stores playback-ready audio in a queue, to be used by a downstream application as an iterator. This is
        useful for applications that benefit from accessing the generated audio in a non-blocking way (e.g. in an interactive
        Gradio demo).
        Parameters:
            model (`MusicgenForConditionalGeneration`):
                The MusicGen model used to generate the audio waveform.
            device (`str`, *optional*):
                The torch device on which to run the computation. If `None`, will default to the device of the model.
            play_steps (`int`, *optional*, defaults to 10):
                The number of generation steps with which to return the generated audio array. Using fewer steps will 
                mean the first chunk is ready faster, but will require more codec decoding steps overall. This value 
                should be tuned to your device and latency requirements.
            stride (`int`, *optional*):
                The window (stride) between adjacent audio samples. Using a stride between adjacent audio samples reduces
                the hard boundary between them, giving smoother playback. If `None`, will default to a value equivalent to 
                play_steps // 6 in the audio space.
            timeout (`int`, *optional*):
                The timeout for the audio queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
                in `.generate()`, when it is called in a separate thread.
        """
        self.music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", torch_dtype=torch.float16).eval()
        self.music_processor = MusicgenProcessor.from_pretrained("facebook/musicgen-small")

        self.decoder = self.music_model.decoder
        self.audio_encoder = self.music_model.audio_encoder
        self.generation_config = self.music_model.generation_config
        self.device = device if device is not None else model.device

        # variables used in the streaming process
        self.play_steps = play_steps
        if stride is not None:
            self.stride = stride
        else:
            hop_length = np.prod(self.audio_encoder.config.upsampling_ratios)
            self.stride = hop_length * (play_steps - self.decoder.num_codebooks) // 6
        self.token_cache = None
        self.to_yield = 0

        # varibles used in the thread process
        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def apply_delay_pattern_mask(self, input_ids):
        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to MusicGen)
        _, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1],
        )
        # apply the pattern mask to the input ids
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, decoder_delay_pattern_mask)

        # revert the pattern delay mask by filtering the pad token id
        input_ids = input_ids[input_ids != self.generation_config.pad_token_id].reshape(
            1, self.decoder.num_codebooks, -1
        )

        # append the frame dimension back to the audio codes
        input_ids = input_ids[None, ...]

        # send the input_ids to the correct device
        input_ids = input_ids.to(self.audio_encoder.device)

        output_values = self.audio_encoder.decode(
            input_ids,
            audio_scales=[None],
        )
        audio_values = output_values.audio_values[0, 0]
        return audio_values.cpu().float().numpy()

    def put(self, value):
        batch_size = value.shape[0] // self.decoder.num_codebooks
        if batch_size > 1:
            raise ValueError("MusicgenStreamer only supports batch size 1")

        if self.token_cache is None:
            self.token_cache = value
        else:
            self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)

        if self.token_cache.shape[-1] % self.play_steps == 0:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
            self.on_finalized_audio(audio_values[self.to_yield : -self.stride])
            self.to_yield += len(audio_values) - self.to_yield - self.stride

    def end(self):
        """Flushes any remaining cache and appends the stop symbol."""
        if self.token_cache is not None:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
        else:
            audio_values = np.zeros(self.to_yield)

        self.on_finalized_audio(audio_values[self.to_yield :], stream_end=True)

    def on_finalized_audio(self, audio: np.ndarray, stream_end: bool = False):
        """Put the new audio in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.audio_queue.put(audio, timeout=self.timeout)
        if stream_end:
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get(timeout=self.timeout)
        if not isinstance(value, np.ndarray) and value == self.stop_signal:
            raise StopIteration()
        else:
            return value
    def infer(text_prompt, audio_length_in_s=10.0, play_steps_in_s=2.0, seed=0):
        sampling_rate = self.music_model.audio_encoder.config.sampling_rate
        frame_rate = self.music_model.audio_encoder.config.frame_rate
        target_dtype = np.int16
        max_range = np.iinfo(target_dtype).max
        max_new_tokens = int(frame_rate * audio_length_in_s)
        play_steps = int(frame_rate * play_steps_in_s)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device != model.device:
            model.to(device)
            if device == "cuda:0":
                model.half()

        inputs = self.music_processor(
            text=text_prompt,
            padding=True,
            return_tensors="pt",
        )
        streamer = MusicgenStreamer(model, device=device, play_steps=play_steps)
        generation_kwargs = dict(
            **inputs.to(device),
            streamer=streamer,
            max_new_tokens=max_new_tokens,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        set_seed(seed)
        for new_audio in streamer:
            print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 2)} seconds")
            new_audio = (new_audio * max_range).astype(np.int16)
            yield sampling_rate, new_audio
