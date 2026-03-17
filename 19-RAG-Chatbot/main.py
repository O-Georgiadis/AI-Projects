import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr
from pathlib import Path

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = Path(__file__).parent / "vector_db"


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)



SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

def answer_question(question: str, history):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content


# print(answer_question("Who is Avery?", []))


gr.ChatInterface(answer_question).launch(inbrowser=True)