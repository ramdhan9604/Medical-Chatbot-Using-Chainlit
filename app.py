import chainlit as cl
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from src.prompt import system_prompt

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone vector store
index_name = "medicalbot2"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-specdec", temperature=0.8)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create retrieval-augmented generation (RAG) pipeline
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Chainlit chat handler
@cl.on_message
async def chat(message):
    print("User input:", message.content)
    response = rag_chain.invoke({"input": message.content})
    print("Response:", response["answer"])
    await cl.Message(content=response["answer"]).send()

if __name__ == '__main__':
    cl.run()
