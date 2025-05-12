# IMPORTING LIBRARIES
import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings

# Suppress warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# ENVIRONMENT VARIABLE TO AVOID ERROR MESSAGES
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# DEFINE PERSISTENT DIRECTORY FOR CHROMADB
PERSIST_DIRECTORY = os.path.join("../data/", "Vectors")

# PAGE CONFIG AND STYLE
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    .stChatMessage {
        padding: 10px 15px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 Chat with your PDF.")
st.markdown("""This a Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama and LangChain that enables users to upload a PDF document, extract and process its contents, and ask context-aware questions based on the material. It leverages a selected language model via Ollama and LangChain to generate accurate, retrieval-enhanced responses grounded in the document's information.""")

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# MODEL SELECTION
def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.
    """
    logger.info("Extracting model names from models_info")
    try:
        if hasattr(models_info, "models"):
            # model_names = tuple(model.model for model in models_info.models)

            # Filter out specific models by name
            exclude_models = {"mxbai-embed-large:latest", "nomic-embed-text:latest"}
            model_names = tuple(
                model.model for model in models_info.models 
                if model.model not in exclude_models
            )

        else:
            model_names = tuple()
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()

def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=f"pdf_{hash(file_upload.name)}"
    )
    logger.info("Vector DB created with persistent storage")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    llm = ChatOllama(model=selected_model)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            vector_db.delete_collection()
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")

def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    # Sidebar layout
    st.sidebar.header("Upload PDF & Model Configuration")
    # st.sidebar.subheader("Upload PDF & Model Configuration")
    col1, col2 = st.columns([1, 2])

    # Sidebar options
    with st.sidebar:
        models_info = ollama.list()
        available_models = extract_model_names(models_info)
        selected_model = st.selectbox(
            "Select a model from the list below", 
            available_models,
            key="model_select"
        )

        use_sample = st.checkbox("Use Sample PDF", key="sample_checkbox")
    
    # Handle file upload
    if use_sample:
        # sample_path = "../data/PDFs/Sustainable_development_of_distance_learning_in_continuing_adult_education__The impact_of_artificial_intelligence.pdf"

        sample_path = os.path.join(os.path.dirname(__file__), '..', 'data','PDFs', 'Sustainable_development_of_distance_learning_in_continuing_adult_education__The impact_of_artificial_intelligence.pdf')


        if os.path.exists(sample_path):
            if st.session_state.get("vector_db") is None:
                with st.spinner("Processing sample PDF..."):
                    loader = UnstructuredPDFLoader(file_path=sample_path)
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                    chunks = text_splitter.split_documents(data)
                    st.session_state["vector_db"] = Chroma.from_documents(
                        documents=chunks,
                        embedding=OllamaEmbeddings(model="nomic-embed-text"),
                        persist_directory=PERSIST_DIRECTORY,
                        collection_name="sample_pdf"
                    )
                    with pdfplumber.open(sample_path) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
        else:
            st.error("Sample PDF not found.")
    else:
        file_upload = st.file_uploader("Upload PDF", type="pdf")
        if file_upload:
            if st.session_state.get("vector_db") is None:
                with st.spinner("Processing PDF..."):
                    st.session_state["vector_db"] = create_vector_db(file_upload)
                    with pdfplumber.open(file_upload) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]

    # Display PDF
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        zoom_level = st.slider("Zoom Level", min_value=100, max_value=1000, value=700, step=50)
        st.image(st.session_state["pdf_pages"], width=zoom_level)

    # Delete collection button
    if st.button("⚠️ Delete Collection"):
        delete_vector_db(st.session_state.get("vector_db"))

    # Chat interface
    st.subheader("💬 Chat with the PDF Content")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if prompt := st.chat_input("Ask a question..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.spinner("Processing..."):
            response = process_question(prompt, st.session_state.get("vector_db"), selected_model)
        st.session_state["messages"].append({"role": "assistant", "content": response})

    for msg in st.session_state["messages"]:
        avatar = "🤖" if msg["role"] == "assistant" else "😎"
        st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])

if __name__ == "__main__":
    main()
