import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

OPENAI_MODEL = "gpt-4.1-mini"

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found")

openai = OpenAI()

system_message = """
You are a helpful assistant in a bed and mattress store.

You should gently encourage customers to try beds and mattresses that are on sale.
King-size beds are 60% off, and most other beds and mattresses are 50% off.

For example, if a customer says "I'm looking to buy a bed", you could reply:
"Wonderful — we have many comfortable beds available, including several that are part of our sales event."

Encourage customers to consider king-size beds if they are unsure what to get.

If the customer asks for sofas or other furniture, explain that those items are not on sale today,
but remind them to take a look at our beds and mattresses instead.
"""

def chat(message, history):
    history = [
        {"role":h["role"], "content":h["content"]} for h in history
    ]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(
        model= OPENAI_MODEL,
        messages= messages,
        stream= True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result



gr.ChatInterface(fn=chat).launch(inbrowser=True)


