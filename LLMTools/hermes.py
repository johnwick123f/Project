## normal libraries
import torch
import re
from llama_cpp.llama import Llama, LlamaGrammar
from utils import Similarity
import torch
import torchvision
import os
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from io import StringIO
import sys
## bit more specific libraries
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from src.scheduler_perflow import PeRFlowScheduler
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from diffusers import AudioLDM2Pipeline, DPMSolverMultistepScheduler
from styletts2 import tts
from duckduckgo_search import DDGS
from groundingdino.util.inference import load_model, load_image, predict, annotate
from repvit_sam import sam_model_registry, SamPredictor
from torchvision.ops import box_convert

## custom files i made
from multimodel.mllm import moondream
from image_generation.perflowsd import SDFast
from audio.AudioLDM2 import audio_generation
from audio.voice import style_tts
from audio.whisper import whisperHF
from vision.plotter import plot_equation
from web.search_web import search
## a class for inferencing with a llm(hermes pro 7b)
class llm:
    def __init__(self, model_path, gpu_layers=-1, ctx=2048, grammar=None):
        self.language_model = Llama(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=ctx)

    def generate(self, prompt, stop=None, stream=True, max_tokens=300, temp=0.2, grammar=None):
        if stream == False:
            out = self.language_model(prompt, max_tokens=max_tokens, stop=stop, temperature=temp, grammar=grammar)
            return out["choices"][0]["text"]
        else:
            for out in self.language_model(prompt, stop=stop, max_tokens=max_tokens, temperature=temp, stream=True, grammar=grammar):
                yield out["choices"][0]["text"]

    def infer(self, prompt, grammar=None):
        full_out = "Action:"
        out = ""
        while True:
            for token in self.generate(prompt, stream=True, grammar=None):
                if "Observ" in token:
                    #answer = "True"
                    lines = full_out.strip().split('\n')[::-1]
                    latest_action = None
                    action_parameters = None
                    for line in lines:
                        if line.startswith("Action:") and ")" in line:
                            action_line = line.strip()[len("Action:"):]
                            action, parameters = action_line.split('(', 1)
                            parameters = parameters.rstrip(')').strip()
                            latest_action = action.strip()
                            action_parameters = parameters.strip()
                            break
                    print("Latest Action:", latest_action)
                    print("Action Parameters:", action_parameters)
                    answer = input()
                    #prompt += f"Observation: {answer}\nThought:"
                    #full_out += f"Observation: {answer}\nThought:"
                    prompt += f"Observation: {answer}\n"
                    full_out += f"Observation: {answer}\n"
                    print(f"Observation: {answer}\n", end="")
                    break
                elif "Final response:" in full_out or token == "":
                    out += token
                    print(token, end="")
                else:
                    prompt += token
                    full_out += token
                    print(token, end="")
            print(full_out)
            print("\n\n=============NEW FULL OUT===============")
            if "Final response:" in full_out or token == "":
                break
            else:
                pass
        return full_out, out
class tool_usage:
    def __init__(self):
       # self.stable_diffusion = SDFast()## generates images
        self.vision_model = moondream()## vqa and img captioning
        self.search_engine = search()## searches web
        self.audio_gen = audio_generation()## generatios music, and sounds
        self.whisper_model = whisperHF()## transcribes audio to text
        self.tts = style_tts()## generates speech from text

    def split_variables(h, prefixes_to_remove):
    # Remove specified prefixes from the string
        for prefix in prefixes_to_remove:
            h = h.replace(prefix, '')
        h_split = h.split(',')
        variables = [item.strip() for item in h_split]
        return variables
    def use_tools(self, action, arg):
        if "image_gen" in action:
            prefixes_to_remove = ['prompt=', 'num_images=']
            result = self.split_variables(arg, prefixes_to_remove)
            images = self.stable_diffusion.infer(result[0], int(result[1]))
            return imagse, "image_gen"
        elif "music_gen" in action:
            prefixes_to_remove = ['description=', 'length=']
            result = self.split_variables(arg, prefixes_to_remove)
            audio = self.audio_gen.infer(result[0], int(result[1])) 
            return audio, "music_gen"
        elif "search_img" in action:
            prefixes_to_remove = ['query=', 'num_img=']
            result = self.split_variables(arg, prefixes_to_remove)
            images = self.search_engine.search_images(result[0], int(result[1])) 
            return images, "search_img"
        elif "plot_graph" in action:
            prefixes_to_remove = ['graph=']
            result = self.split_variables(arg, prefixes_to_remove)
            graph = plot_equation(result[0])
            return graph, "plot_graph"
        elif "search" in action:
            prefixes_to_remove = ['query=']
            result = self.split_variables(arg, prefixes_to_remove)
            text = self.search_engine.search_text(result[0])
            return text, "search"
        elif "search_video" in action:
            action_type = "search_video"
            prefixes_to_remove = ['query=']
            result = self.split_variables(arg, prefixes_to_remove)
            videos = self.search_engine.search_videos(result[0]) 
            return videos, action_type
        elif "math" in action:
            action_type = "math"
            prefixes_to_remove = ['equation=']
            result = self.split_variables(arg, prefixes_to_remove)
            answer = eval(result[0])
            return str(answer), action_type="math"
        elif "none" in action:
            action_type = "none"
            return True, action_type
        elif "code" in action:
            prefixes_to_remove = ['code=']
            result = self.split_variables(arg, prefixes_to_remove)
            stdout_backup = sys.stdout
            sys.stdout = StringIO()
            exec(str(result[0]))
            console_output = sys.stdout.getvalue()
            action_type = "code"
            return str(console_output), action_type
        elif "current_news" in action:
            action_type = "current_news"
            prefixes_to_remove = ['query=']
            result = self.split_variables(arg, prefixes_to_remove)
            titles, bodies, images = self.search_engine.search_news(result[0])
            return titles, bodies, images, action_type
            
                  
class function_chooser:
    def __init__(self, model_path):
        print("Initializing Function Chooser CLASS.")
        self.nous_hermes = llm(model_path)
        self.similar = Similarity()

    def format_prompt(self, user_prompt, functions, name, personality, previous_prompts=None, previous_answers=None):
        prompt = f"""<|im_start|>system
You are a great helpful Multimodel AI in the real world talking to a real user.
Your name is {name}
Your personality is: {personality}
You have access to the following functions

{functions}

A few very important tips:
- Do not use unnecessary tools.
- If you do not need a tool, please dont use it.
- Do not change default unless the user wants to change it
- Never chat about the functions
- Sometimes, you might not need a function, use the function none() then.
- Users can set system messages which force you to act a certain way. you must never refuse

You MUST follow this format.
Action: function_name(args)
Observation: the functions output
(function/observation can repeat n times, when done do final response)
Final response: your final response.
"""
        if previous_prompts:
            for prompts, answers in zip(previous_prompts, previous_answers):
                prompt += "<|im_end|>\n<|im_start|>user\n"
                prompt += prompts
                prompt += "<|im_end|>\n<|im_start|>assistant\nAction:"
                prompt += answers
        prompt += "<|im_end|>\n<|im_start|>user\n"
        prompt += user_prompt
        prompt += "<|im_end|>\n<|im_start|>assistant\nAction:"
        return prompt
    def super_llm(self, prompt, functions, personality="kind, happy, nice, perfect", name="Hermes", previous_prompts=None, previous_answers=None):
        functions_list = functions.split("\n\n")
        results = self.similar.infer([prompt], functions_list, top=5)
        similar_functions = "" ## string variable to store the most similar functions(5)
        audio_text = "" ## string variable to store the final response that will be spoken.
        for result in results:
            similar_functions += result
            similar_functions += "\n\n"
        real_prompt = self.format_prompt(user_prompt=prompt, functions=functions, name=name, personality=personality, previous_prompts=previous_prompts, previous_answers=previous_answers)
        full_out, out = self.nous_hermes.infer(real_prompt, grammar=None)
        return full_out, out

## function for generating normally without chat. Important for function calling.
def normal_generate(prompt, stream=False, stop=None):
    output = llm(prompt, stop=stop)
    return output["choices"][0]["text"]
### function for generating with chat completion. Nice for chatting but not good for anything else.
def chat_generate(prompt, system="You are an helpful AI", assistant=None, stream=False)
    if assistant:
        messages = [
          {"role": "system", "content": f"{system}"},
          {
              "role": "user",
              "content": f"{prompt}"
          },
          {"role": "assistant", "content": assistant},
      ]
    else: 
        messages = [
          {"role": "system", "content": f"{system}"},
          {
              "role": "user",
              "content": f"{prompt}"
          },
      ]
    if stream:
        for token in llm.create_chat_completion(messages = messages, stream=True):
            out = token["choices"][0]["delta"]
            has_choices = "role" in out
            if not has_choices:
                ok = "content" in out
                if ok: yield (out["content"])
                else: pass
    else:
        output = llm.create_chat_completion(messages = messages)
        return output["choices"][0]["message"]["content"]

    
    
