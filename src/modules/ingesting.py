# This file is meant to load, and process the PDF
# importing libraries
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


def loading_PDF(local_path):
    '''LOADING PDF'''
    if local_path:
            print('Loading the document...')
            loader = UnstructuredPDFLoader(file_path=local_path)
            data = loader.load()
            print(f"PDF loaded successfully")
            return data
    else:
        print("Upload a PDF file")


def chunking(pdf):
    '''CHUNKING THE PDF'''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pdf)
    print(f"The chunking process is complete with a final number of chunks of {len(chunks)}")
    return chunks

def data_base(chunks):
    '''CREATE VECTOR DATABASE'''
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="local-rag"
    )
    print("Vector database created successfully")
    return vector_db

