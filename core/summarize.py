from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda

import os

def get_llm ():
    return ChatMistralAI(model="mistral-small-latest", temperature=0.3, mistral_api_key=os.getenv("MISTRAL_API_KEY"))

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(transcript)
