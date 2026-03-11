from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc
import os
from dotenv import load_dotenv


load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)


GEMMA = "google/gemma-3-270m-it"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" #reasoning model


messages = [
    {"role": "user", "content": "Tell a clever joke"}
  ]


# Quantization Config - this allows to load the model into memory and use less memory
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
#=========================== EXPERIMENTING ===========================================================

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(GEMMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("mps")

# The model
model = AutoModelForCausalLM.from_pretrained(GEMMA, device_map="auto", quantization_config=quant_config)

memory = model.get_memory_footprint()/1e6
print(f"Memory footprint: {memory:,.1f} MB")

print(model)
attention_mask = torch.ones_like(inputs)
outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=80, do_sample=False)
print(outputs[0])

tokenizer.decode(outputs[0])

# Clean up memory
del model, inputs, tokenizer, outputs
gc.collect()
torch.mps.empty_cache()

#======================================================================================

# Wrapping everything in a function 

def generate(model, messages, quant=True, max_new_tokens=80):
  tokenizer = AutoTokenizer.from_pretrained(model)
  tokenizer.pad_token = tokenizer.eos_token
  input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("mps")
  attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="mps")
  streamer = TextStreamer(tokenizer)
  if quant:
    model = AutoModelForCausalLM.from_pretrained(model, quantization_config=quant_config).to("mps")
  else:
    model = AutoModelForCausalLM.from_pretrained(model).to("mps")
  outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, streamer=streamer)


generate(GEMMA, messages, quant=False)
generate(DEEPSEEK, messages, quant=False, max_new_tokens=500) 