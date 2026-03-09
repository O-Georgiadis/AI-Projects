from transformers import AutoTokenizer
from huggingface_hub import login
import os
from dotenv import load_dotenv



load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")


if not hf_token:
    raise ValueError("No Hugging Face token found")


login(hf_token, add_to_git_credential=True)

text = "I am experimenting with tokenizers"

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a clever joke"}
  ]


#======================== Llama =======================================
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", trust_remote_code=True)
tokens = tokenizer.encode(text)
print(tokens)

character_count  = len(text)
word_count = len(text.split(" "))
token_count = len(tokens)
print(f"There are {character_count} characters, {word_count} words, and {token_count} tokens")


decoded_tokens = tokenizer.decode(tokens)
print(decoded_tokens)

print(tokenizer.batch_decode(tokens))

# to see all the special tokens (e.g. begin_of_text, end_of_text)
# print(tokenizer.get_added_vocab())

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)


prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)


print("=========="*10)
#======================== Phi4 =======================================

PHI4 = "microsoft/Phi-4-mini-instruct"
phi4_tokenizer = AutoTokenizer.from_pretrained(PHI4)
print(phi4_tokenizer.encode(text))
print(phi4_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

print("=========="*10)
#======================== Deepseek =======================================

DEEPSEEK = "deepseek-ai/DeepSeek-V3.1"

deepseek_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK)
print(deepseek_tokenizer.encode(text)) 
print(deepseek_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


print("=========="*10)
#======================== Qwen =======================================
QWEN_CODER = "Qwen/Qwen2.5-Coder-7B-Instruct"

qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_CODER)
code = """
def hello_world(person):
    print('Hello', person)
"""

tokens = qwen_tokenizer.encode(code)
for token in tokens:
    print(f"{token}: {qwen_tokenizer.decode(token)}")

