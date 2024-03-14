def form(prompt, format=True):
  system = """Try to solve the question you will be given with a single tool.
You have access to the following tools:
Robot
- Description: can do any manipulation, grasping, placing task from a prompt. Not for questions or descriptions
- Returns: None
- Input: prompt

StableDiffusion
- Description: Generates a image from a prompt. Not for anything else
- Returns: None
- Input: prompt

VisualLLM
- Description: Analyzes images with precision, offering insights and understanding effortlessly. Can describe or answer questions about visual things.
- Returns: None
- Input: prompt

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

ONLY use tools that are given. NEVER use made up tools.
ALWAYS use the following format:

Thought: you should always think about what to do
Action:
```
{{{{
  "Action": $TOOL_NAME,
  "Action_input": $INPUT
}}}}
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
  if format:
    full_prompt = f"""
  <|im_start|>system
{system}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
Thought: I need to clarify the request and provide an appropriate response using the available tools.
Action: 
```
{{{{{{{{
  "Action": """ + "\""
    return full_prompt
  else: return prompt, system
