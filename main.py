from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o-mini")

if __name__ == "__main__":
    print("Hello")
    pdf_path = "./Generative-AI-and-LLMs-for-Dummies.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    vector_store.save_local("faiss_index")

    new_vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vector_store.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "What is the capital of France and Indonesia?"})
    print(res["answer"])

