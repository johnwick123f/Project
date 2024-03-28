from llama_cpp import Llama
from LLMTools.prompts import main_llm_prompt
class llm:
    def __init__(self, model_path, gpu_layers=-1, ctx=2048):
        self.language_model = llm = Llama(model_path=model_path, n_gpu_layers=gpu_layers n_ctx=ctx)

    def generate(self, prompt, stop, stream=True, max_tokens=300, temp=0.2):
        if stream == False:
            out = llm(prompt, max_tokens=max_tokens, stop=stop, temperature=temp)
            return out["choices"][0]["text"]
        else:
            for out in llm(prompt, max_tokens=max_tokens, stop=stop, temperature=temp, stream=True):
                yield out["choices"][0]["text"]

    def infer(self, prompt):
        full_out = ""
        out = ""
        while True:
            for token in normal_generate(prompt, stream=True):
                if "Observ" in token: 
                    answer = "True"
                    prompt += f"Observation: {answer}\nThought:"
                    full_out += f"Observation: {answer}\nThought:"
                    break
                elif "Final Answer:" in full_out:
                    out += token
                else:
                    prompt += token
                    full_out += token
                    #print(token, end="")
            if "Final Answer" in full_out: 
                break
            else:
                pass

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

    
    
