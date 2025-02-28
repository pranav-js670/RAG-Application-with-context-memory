from dotenv import load_dotenv
import os
load_dotenv()
USER_AGENT = os.environ.get('USER_AGENT')
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import StrOutputParser

def get_docs():
    loader = WebBaseLoader(["https://my.clevelandclinic.org/health/diseases/16818-heart-attack-myocardial-infarction",
                            "https://my.clevelandclinic.org/health/diseases/21493-cardiovascular-disease"])
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splitDocs = text_splitter.split_documents(docs)
    return splitDocs

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def create_chain(vector_store):
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )

    contextualize_q_system_prompt = """
    Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a standalone question which can be understood
    without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    contextualize_chain = contextualize_q_prompt | model | StrOutputParser()

    retriever = vector_store.as_retriever()

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    print(f'Retrieval chain created')
    return rag_chain

def process_chat(chain, user_input, chat_history):
    response = chain.invoke({"input": user_input, "chat_history": chat_history})['answer']
    
    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=response)
    ])
    
    return response

if __name__ == "__main__":
    docs = get_docs()
    vector_store = create_vector_store(docs)
    chain = create_chain(vector_store)

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = process_chat(chain, user_input, chat_history)
        print("Assistant:", response)
        print()
