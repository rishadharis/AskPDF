from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, please output "Sorry i don't know", don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "Thanks, Terima Kasih, Kamsahamnida!" at the end of your answer.

    Context:
    ```
    {context}
    ```

    Question: `{question}`

    Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    result = rag_chain.invoke("What is LLM?")
    print(result)

