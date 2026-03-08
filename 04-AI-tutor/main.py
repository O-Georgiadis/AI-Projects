from openai import OpenAI

MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"



def build_messages(code):

    system_prompt = "You are a helpful technical tutor who answers questions about python code, software engineering, data science and LLMs"
    user_prompt = "Please give a detailed explanation of what this code does and why: " + code

    return  [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

def get_explanation(code):
    ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama_key")
    response = ollama.chat.completions.create(
        model=MODEL,
        messages = build_messages(code)
    )
    print(response.choices[0].message.content)

def main():
    code = input("Please insert code to be explained: ")
    print("\nProcessing Code...\n")
    print(get_explanation(code))
    


if __name__ == "__main__":
    main()

