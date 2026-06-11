from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import os

def get_llm ():
    return ChatMistralAI(model="mistral-small-latest", temperature=0.3, mistral_api_key=os.getenv("MISTRAL_API_KEY"))

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages(
        ("system", "You are a helpful assistant that summarizes meeting transcripts."),
        ("human", "{text}")
    )
    
    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            "messages={summaries}\n\nBased on the above summaries, provide a concise summary of the entire meeting. Focus on key points, decisions, and action items." 
        ),
        ("human", "{text}")
        ]
    )
   
    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text":x}) |combined_prompt | llm | StrOutputParser()
        )
    return combined_chain.invoke(combined)

def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text":x}) | ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that generates concise titles for meeting transcripts. Maximum 8 words only and avoid using generic words like 'Meeting' or 'Discussion'. Focus on the main topic or key outcome of the meeting."),
                ("human", "{text}\n\nBased on the above transcript, generate a concise title that captures the main topic of the meeting.")
            ]
        ) | llm | StrOutputParser()
    )
    return title_chain.invoke({transcript[:2000]})
