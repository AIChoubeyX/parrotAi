# Actions , decisions and key points extractor from meeting transcripts using Mistral-7B-Instruct-v0.1-Q4_0.gguf

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os

from requests import get

def get_llm():
    return ChatMistralAI(model="mistral-small-latest", temperature=0.3, mistral_api_key=os.getenv("MISTRAL_API_KEY"))

def build_chain(system_prompt : str) :
    llm = get_llm()
    chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text":x}) | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}")
            ]
        ) | llm | StrOutputParser()
    )
    return chain

def extract_action_items(transcript: str) -> str:
    chain = build_chain("You are a helpful assistant that extracts action items from meeting transcripts. Focus on identifying specific tasks, who is responsible for each task, and any deadlines mentioned.")

    return chain.invoke(transcript)

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain("You are a helpful assistant that extracts key decisions from meeting transcripts. Focus on identifying important choices made during the meeting, including the context and rationale behind each decision.")

    return chain.invoke(transcript)

def extract_questions(transcript: str) -> str:
    chain = build_chain("You are a helpful assistant that extracts questions from meeting transcripts. Focus on identifying any questions raised during the meeting, including who asked the question and any relevant context.")

    return chain.invoke(transcript)