import os
from scraper import fetch_website_contents
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2"
OPENAI_MODEL = "gpt-4.1-mini"



load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    print(f"OpenAI API Key exists")
else:
    print("OpenAI API Key not set")

    
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
openai = OpenAI()


system_message = """
You are an assistant that analyzes the contents of a company website landing page
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
"""

#-----------------GPT------------------------------------------------

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


#-----------------------------------------------------------------------------------



def stream_brochure(company_name, url, model):
    yield ""
    prompt = f"Generate a company brochure for {company_name}. Their landing page is:\n"
    prompt += fetch_website_contents(url)
    if model=="GPT":
        result = message_gpt(prompt)
    elif model=="Ollama":
        result = message_ollama(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result


name_input = gr.Textbox(label="Company name:", lines=1)
url_input = gr.Textbox(label="Landing page URL including http:// or https://", lines=1)
model_selector = gr.Dropdown(["GPT", "Ollama"], label="Select model", value="GPT")
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_brochure,
    title="Brochure Generator", 
    inputs=[name_input, url_input, model_selector], 
    outputs=[message_output], 
    examples=[
            ["Hugging Face", "https://huggingface.co", "GPT"],
            ["Hugging Face", "https://huggingface.co", "Ollama"]
        ], 
    flagging_mode="never"
    )
view.launch(inbrowser=True)