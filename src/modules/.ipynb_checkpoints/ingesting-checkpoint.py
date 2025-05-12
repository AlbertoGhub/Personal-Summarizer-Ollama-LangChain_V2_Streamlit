# This file is meant to load, and process the PDF
# importing libraries
from langchain_community.document_loaders import UnstructuredPDFLoader

def loading_PDF(local_path):
    '''TO LOAD PDF'''
            
    if local_path:
            print('Loading the document...')
            loader = UnstructuredPDFLoader(file_path=local_path)
            data = loader.load()
            print(f"PDF loaded successfully")
            return data
    else:
        print("Upload a PDF file")

