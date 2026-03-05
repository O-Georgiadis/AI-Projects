import os
from dotenv import load_dotenv
from openai import OpenAI

import gradio as gr

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2"
OPENAI_MODEL = "gpt-4.1-mini"
DEMO_USERNAME = "od"
DEMO_PASSWORD = "1234"

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')


if openai_api_key:
    print(f"OpenAI API Key exists")
else:
    print("OpenAI API Key not set")



ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
openai = OpenAI()


system_message = "You are a helpful assistant"

#-----------------GPT------------------------------------------------

message_input = gr.Textbox(label="Your message:", info="Enter a message for the Gpt-4.1-mini", lines=7)
message_output = gr.Markdown(label="Response:")

def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result



#----------------OLLAMA--------------------------------------------------------------

def message_ollama(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = ollama.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


message_input = gr.Textbox(label="Your message:", info="Enter a message for llama 3.2", lines=7)
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=message_ollama,
    title="Ollama", 
    inputs=[message_input], 
    outputs=[message_output], 
    examples=[
        "Explain the Transformer architecture to a layperson",
        "Explain the Transformer architecture to an aspiring AI engineer",
        ], 
    flagging_mode="never"
    )
view.launch(inbrowser=True, auth=(DEMO_USERNAME, DEMO_PASSWORD))

#--------------------------------------------------------------------------------------------------

def stream_model(prompt, model):
    if model=="GPT":
        result = message_gpt(prompt)
    elif model=="Ollama":
        result = message_ollama(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result



message_input = gr.Textbox(label="Your message:", info="Enter a message for the LLM", lines=7)
model_selector = gr.Dropdown(["GPT", "Ollama"], label="Select model", value="GPT")
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_model,
    title="LLMs", 
    inputs=[message_input, model_selector], 
    outputs=[message_output], 
    examples=[
            ["Explain the Transformer architecture to a layperson", "GPT"],
            ["Explain the Transformer architecture to an aspiring AI engineer", "Ollama"]
        ], 
    flagging_mode="never"
    )
view.launch()