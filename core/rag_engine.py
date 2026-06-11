import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough , RunnableLambda
from core.vector_store import create_vector_store,load_vector_store, get_retriever

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])
        
def build_rag_chain(transcript: str):
    vector_store = create_vector_store(transcript)
    retriever = get_retriever(vector_store)
    llm = get_llm()
    
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system", 
            """You are a expert meeting assistant . Answer the question based on the provided context from the meeting transcript. If the answer is not in the context, say you don't know. Always be concise and to the point.
            context from meeting transcript : {context}
            """
            ),
        ("human", " following question:\n{question}")
    ])
    
    rag_chain = (

        {"context": retriever | RunnableLambda(format_docs), 
         "question": RunnablePassthrough()
         }
         | prompt_template | llm | StrOutputParser()
    )
    return rag_chain

def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    llm = get_llm()
    
    prompt_template = ChatPromptTemplate.from_message ([
        (
            "system", 
            """You are a expert meeting assistant . Answer the question based on the provided context from the meeting transcript. If the answer is not in the context, say you don't know. Always be concise and to the point.
            context from meeting transcript : {context}
            """
            ),
        ("human", " following question:\n{question}")
    ])
    
    rag_chain = (

        {"context": retriever | RunnableLambda(format_docs), 
         "question": RunnablePassthrough()
         }
         | prompt_template | llm | StrOutputParser()
    )
    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke({"question": question})
    print(f"Answer: {answer}")
    return answer