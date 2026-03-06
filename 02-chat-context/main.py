import tiktoken 
from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.2"

# Transform Text to Tokens
encoding = tiktoken.encoding_for_model("gpt-4.1-mini")

tokens = encoding.encode("Hi my name is Odysseas")
print(tokens)

# Transform Tokens to Text
for token_id in tokens:
    token_text = encoding.decode([token_id])
    print(f"{token_id} - {token_text}")

ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hi, I'm Odysseas"}
]

response = ollama.chat.completions.create(model=MODEL, messages=messages)
print(response.choices[0].message.content)

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hi, I'm Odysseas"},
    {"role": "assistant", "content":"A legendary name! I'm delighted to make your acquaintance, Odysseus. What can I do for you today? Are you a king on a quest, or perhaps just seeking some advice or guidance?"},
    {"role": "user", "content": "What's my name?"}
]

response = ollama.chat.completions.create(model=MODEL, messages=messages)
print(response.choices[0].message.content)