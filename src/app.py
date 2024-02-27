from typing import List

from src.constants import PROMPT_TEMPLATE

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

# Vector embedding and vector store
from langchain_community.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def get_embedding_model(model_id: str, client):
    # Bedrock client
    bedrock_embeddings = BedrockEmbeddings(model_id=model_id, client=client)
    return bedrock_embeddings


def data_ingestion(data_path: str, chunk_size: int = 10000, chunk_overlap: int = 1000) -> List[Document]:
    loader = PyPDFDirectoryLoader(data_path)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(document)
    return docs


# Vector embeddings and vector store
def get_vector_store(docs: List[Document], embedding_model: BedrockEmbeddings, save_path: str) -> str:
    vectorstore_faiss = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model
    )
    vectorstore_faiss.save_local(save_path)


def get_llama2_llm(model_id: str, client, model_kwargs):
    # create the Anthropic Model
    llm = Bedrock(model_id=model_id, client=client,
                  model_kwargs=model_kwargs)
    return llm


def get_response_llm(llm, vectorstore_faiss, query):
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=['context', 'question ']
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    answer = qa({"query": query})
    return answer['result']
