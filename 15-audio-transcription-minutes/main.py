import os
import requests
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, pipeline
import torch
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

GEMMA = "google/gemma-3-270m-it"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" #reasoning model

audio_filename = Path(__file__).parent/"denver_extract.mp3"



#============ Transcription with Open source =======================
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium.en",
    # dtype=torch.float16,
    device='mps',
    return_timestamps=True
)

result = pipe(str(audio_filename))
transcription = result["text"]
# print(transcription)

#================ Transcription with OpenAI ==========================

# openai_api_key = os.getenv('OPENAI_API_KEY')


# if not openai_api_key:
#     raise ValueError("OpenAI API key not found")

# AUDIO_MODEL = "gpt-4o-mini-transcribe"


# openai = OpenAI(api_key=openai_api_key)
# transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio_filename, response_format="text")
# print(transcription)

#========================================================================


system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks.
"""

user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners

Transcription:
{transcription}
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

def generate(model, messages, max_new_tokens=80):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("mps")
    # attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="mps")
    streamer = TextStreamer(tokenizer)
  
    model = AutoModelForCausalLM.from_pretrained(model).to("mps")
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, streamer=streamer)

generate(GEMMA, messages, max_new_tokens=2000)
# generate(DEEPSEEK, messages, max_new_tokens=500) 