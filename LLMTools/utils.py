from llama_cpp import Llama
# loads model
def load_model(model_path, gpu_layers=-1, ctx=2048):
    llm = Llama(
          model_path=model_path,
          n_gpu_layers=gpu_layers
          n_ctx=ctx,
    )
    return llm
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

    
    
