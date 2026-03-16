import os
from dotenv import load_dotenv
from openai import OpenAI
from system_info import retrieve_system_info
from pathlib import Path

OPENAI_MODEL = "gpt-5"

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')


if openai_api_key:
    print(f"OpenAI API Key exists")
else:
    print("OpenAI API Key not set")

openai = OpenAI()

#==================Setting everything up=====================
system_info = retrieve_system_info()



message = f"""
Here is a report of the system information for my computer.
I want to run a C++ compiler to compile a single C++ file called main.cpp and then execute it in the simplest way possible.
Please reply with whether I need to install any C++ compiler to do this. If so, please provide the simplest step by step instructions to do so.

If I'm already set up to compile C++ code, then I'd like to run something like this in Python to compile and execute the code:
```python
compile_command = # something here - to achieve the fastest possible runtime performance
compile_result = subprocess.run(compile_command, check=True, text=True, capture_output=True)
run_command = # something here
run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
return run_result.stdout
```
Please tell me exactly what I should use for the compile_command and run_command.

System information:
{system_info}
"""

response = openai.chat.completions.create(model=OPENAI_MODEL, messages=[{"role": "user", "content": message}])
print(response.choices[0].message.content)
    
compile_command = ["clang++", "-std=c++20", "-O3", "-flto=thin", "-mcpu=native", "-DNDEBUG", "main.cpp", "-o", "main"]
run_command = ["./main"]

#================================================================
language = "C++" # or "Rust"
extension = "rs" if language == "Rust" else "cpp"

system_prompt = f"""
Your task is to convert Python code into high performance {language} code.
Respond only with {language} code. Do not provide any explanation other than occasional comments.
The {language} response needs to produce an identical output in the fastest possible time.
"""

def user_prompt_for(python):
    return f"""
        Port this Python code to {language} with the fastest possible implementation that produces identical output in the least time.
        The system information is:
        {system_info}
        Your response will be written to a file called main.{extension} and then compiled and executed; the compilation command is:
        {compile_command}
        Respond only with {language} code.
        Python code to port:

        ```python
        {python}
        ```
        """


def messages_for(python):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(python)}
    ]
 


def write_output(cpp):
    with open(Path(__file__).parent/f"main.{extension}", "w", encoding="utf-8") as f:
        f.write(cpp)


def port(client, model, python):
    reasoning_effort = "high" if 'gpt' in model else None
    response = client.chat.completions.create(model=model, messages=messages_for(python), reasoning_effort=reasoning_effort)
    reply = response.choices[0].message.content
    reply = reply.replace('```cpp','').replace('```rust','').replace('```','')
    write_output(reply)



test_code = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(200_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""



def run_python(code):
    globals = {"__builtins__": __builtins__}
    exec(code, globals)


run_python(test_code)


port(openai, OPENAI_MODEL, test_code)