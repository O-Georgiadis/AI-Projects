import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from database import price_setter
import sqlite3
import base64
from io import BytesIO
from PIL import Image
import tempfile


OPENAI_MODEL = "gpt-4o-mini"

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')


if not openai_api_key:
    raise ValueError("OpenAI API key not found")

openai = OpenAI()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "prices.db")

# Initialize the database with prices
price_setter()



system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""


# ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(city):
    print(f"DATABASE TOOL CALLED: Getting price for {city}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT price FROM prices WHERE city = ?', (city.lower(),))
        result = cursor.fetchone()
        return f"Ticket price to {city} is ${result[0]}" if result else "No price data available for this city"
    
# The Json describes the function to the LLM
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}


#--------------------------- Image Generator -----------------------------------------------------------#
def create_image(city):
    image_response = openai.images.generate(
            model="dall-e-3",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))
#-----------------------------------------------------------------------------------------------#

#--------------------- Text To Speech -----------------------------------------------------------#
def text_to_speech(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=message
    )
    # Save audio to temporary file and return path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name

#-------------------------------------------------------------------------------------------------#
tools = [{"type": "function", "function": price_function}]


def chat(history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages, tools=tools)
    cities = []
    image = None
    voice = None

    while response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        responses, cities = handle_tool_calls(message)
        messages.append(message)
        messages.extend(responses)
        response = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages, tools=tools)

    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]

    try:
        voice = text_to_speech(reply)
        print(f"Voice generated: {voice}", flush=True)
    except Exception as e:
        print(f"Error generating speech: {e}", flush=True)
        voice = None

    if cities:
        image = create_image(cities[0])
    
    print(f"Returning - Voice: {voice}, Image: {image}", flush=True)
    return history, voice, image


def handle_tool_calls(message):
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            cities.append(city)
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses, cities


#------------------------------ UI --------------------------------------------------------#

def put_message_in_box(message, history):
    return "", history + [{"role": "user", "content": message}]

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500)
        image_output = gr.Image(height=500, interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI assistant:")


    message.submit(put_message_in_box, inputs=[message, chatbot], outputs=[message, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output]
    )

ui.launch(inbrowser=True)