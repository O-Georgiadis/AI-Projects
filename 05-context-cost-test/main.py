from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path
from litellm import completion


OPENAI_MODEL = "gpt-5-mini"


load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')


if openai_api_key:
    print(f"OpenAI API Key exists")
else:
    print("OpenAI API Key not set")



openai = OpenAI()


def cost_and_usage(response):
    print(f"Input tokens: {response.usage.prompt_tokens}")
    print(f"Output tokens: {response.usage.completion_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print(f"Total cost: {response._hidden_params['response_cost']*100:.4f} cents")



with open(Path(__file__).parent/"hamlet.txt", "r", encoding="utf-8") as f:
    hamlet = f.read()

question = [{"role": "user", "content": "In Hamlet, when Laertes asks 'Where is my father?' what is the reply?"}]

response = completion(model="openai/gpt-4.1-mini", messages=question)
print(response.choices[0].message.content)
cost_and_usage(response)

print("\n" * 10)
question[0]["content"] += "\n\nFor context, here is the entire text of Hamlet:\n\n"+hamlet

response = completion(model="gpt-4.1-mini", messages=question)
print(response.choices[0].message.content)
cost_and_usage(response)


