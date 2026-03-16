import os
import glob
from dotenv import load_dotenv
from pathlib import Path
import gradio as gr
from openai import OpenAI


OPENAI_MODEL = "gpt-5"

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')


if openai_api_key:
    print(f"OpenAI API Key exists")
else:
    print("OpenAI API Key not set")

openai = OpenAI()

knowledge = {}

filenames = (Path(__file__).parent/"knowledge-base/employees").glob("*")

for filename in filenames:
    name = Path(filename).stem.split(' ')[-1]
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()

# print(knowledge["lancaster"])


# filenames = (Path(__file__).parent/"knowledge-base/products").glob("*")

# for filename in filenames:
#     name = Path(filename).stem
#     with open(filename, "r", encoding="utf-8") as f:
#         knowledge[name.lower()] = f.read()

# print(knowledge.keys())



SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""


def get_relevant_context(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    return [knowledge[word] for word in words if word in knowledge]   

def additional_context(message):
    relevant_context = get_relevant_context(message)
    if not relevant_context:
        result = "There is no additional context relevant to the user's question."
    else:
        result = "The following additional context might be relevant in answering the user's question:\n\n"
        result += "\n\n".join(relevant_context)
    return result



def chat(message, history):
    system_message = SYSTEM_PREFIX + additional_context(message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return response.choices[0].message.content

view = gr.ChatInterface(chat).launch(inbrowser=True)