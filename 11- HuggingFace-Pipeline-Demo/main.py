import torch
from transformers import pipeline
from diffusers import AutoPipelineForText2Image
from datasets import load_dataset
import soundfile as sf
import subprocess
from huggingface_hub import login
import os
from dotenv import load_dotenv
import gradio as gr
from pathlib import Path

load_dotenv(override=True)


hf_token = os.getenv("HF_TOKEN")


if not hf_token:
    raise ValueError("No Hugging Face token found")


login(hf_token, add_to_git_credential=True)


# Sentiment Analysis

my_simple_sentiment_analyzer = pipeline("sentiment-analysis", device="mps")
result = my_simple_sentiment_analyzer("I'm super excited to be going to work!")
print(result)

result = my_simple_sentiment_analyzer("I should be more excited to be going to work")
print(result)


better_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device="mps")
result = better_sentiment("I should be more excited to be going to work")
print(result)


# Named Entity Recognition

ner = pipeline("ner", device="mps")
result = ner("Im an engineer learning about HuggingFace pipelines with Nick!")
for entity in result:
   print(entity)

# Question Answering with Context

question="What are Hugging Face pipelines?"
context="Pipelines are a high level API for inference of LLMs with common tasks"

question_answerer = pipeline("question-answering", device="mps")

result = question_answerer(question=question, context=context)
print(result)

# Text Summarization

summarizer = pipeline("summarization", device="mps")
text = """
The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP).
It allows users to perform a wide range of tasks such as text classification, named entity recognition, and question answering, among others.
It's an extremely popular library that's widely used by the open-source data science community.
It lowers the barrier to entry into the field by providing Data Scientists with a productive, convenient way to work with transformer models.
"""

summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])

# Translation

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-el", device="mps")
result = translator("Pipelines are a high level API for inference of LLMs with common tasks")
print(result[0]['translation_text'])

# Classification

classifier = pipeline("zero-shot-classification", device="mps")
result = classifier("Hugging Face's Transformers library is amazing!", candidate_labels=["technology", "sports", "politics"])
print(result)


# Text Generation

generator = pipeline("text-generation", device="mps")
result = generator("The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP). It allows users to")
print(result[0]['generated_text'])

# Audio Generation

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='mps')
embeddings_dataset = load_dataset("matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
speech = synthesiser("Hello, I hope your day was perfect today!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write(Path(__file__).parent/"speech.wav", speech["audio"], speech["sampling_rate"])
print("Audio saved as speech.wav")
subprocess.run(["open", "speech.wav"])

# Image Generation

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("mps")
prompt = "An AI engineer coding in front of his computer"
image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
gr.Interface(lambda: image, inputs=[], outputs=gr.Image(type="pil")).launch(inbrowser=True)


