# 📄 Chat with Your PDF

## 🔍 Overview

Welcome to **Chat with Your PDF**! This is an interactive **Streamlit** application that implements **PDF-based Retrieval-Augmented Generation (RAG)** using **Ollama** and **LangChain**. The application allows users to effortlessly upload a PDF document, extract and process its contents, and then ask context-aware questions based on the material. By leveraging a selected language model via Ollama and LangChain, this tool generates precise, retrieval-enhanced responses grounded in the document’s content.

Whether you're working with research papers, technical documents, or any PDF material, this application enables you to extract valuable insights and quickly find answers to your questions directly from the document. It's a powerful tool for anyone looking to interact with their documents intelligently.

---

## ⚙️ Features

- **PDF Upload**: Easily upload your PDF documents to the app.
- **Context-Aware Responses**: Ask questions based on the document, and get accurate, retrieval-enhanced answers.
- **Language Model Selection**: Choose from a variety of models available locally via Ollama.
- **Real-Time Feedback**: Interact in real-time with responses generated from the contents of your uploaded PDF.
- **Document Processing**: The application processes and splits your document into manageable chunks for efficient retrieval.

---

## 🛠️ Technologies Used

- **Streamlit**: For creating the interactive front-end web interface.
- **Ollama**: Used for accessing language models to generate responses.
- **LangChain**: Utilised for RAG operations and document handling.
- **Chroma**: For vector storage and efficient document search.
- **pdfplumber**: To extract images and content from PDFs.

---

## 🚀 Getting Started

To run the **Chat with Your PDF** application locally, follow the steps below.

### 1. Clone the Repository

```bash
git clone https://github.com/AlbertoGhub/Personal-Summarizer-Ollama-LangChain_V2_Streamlit.git
```
### 2. Install Dependencies
- #### Create the virtual environment

```bash
python3 -m venv venv
```
- #### Activate the environment
```bash
source venv/bin/activate  # For MacOS/Linux
venv\Scripts\activate     # For Windows
```

- #### Install the requirements

```bash
pip install -r requirements.txt
````

### 3. Set Up Ollama
To use the models available via Ollama, make sure to have the Ollama environment set up on your machine. Follow the installation guide on the [Ollama website](https://ollama.com/) for instructions on setting up the platform.

### 4. Run the Application
Once the dependencies are installed, you can start the application with the following command:
```bash
streamlit run app.py
```

This will launch the Streamlit interface in your browser, where you can upload your PDF and start interacting with it.

<img width="1439" alt="Image" src="https://github.com/user-attachments/assets/969480c9-5cfc-40a7-bb1f-174c39a665a3" />

### 5. 🧠 How It Works
- **Upload Your PDF:** Select the PDF you wish to interact with. The document will be processed to extract its contents. Enable or disable the sample document with using the check box ```Use sample``` — perfect for trying out the app instantly. 

<img width="1426" alt="Image" src="https://github.com/user-attachments/assets/59da36f3-c7f5-4c51-9c12-4c01a739837d" />

- **Text Splitting:** The PDF content is split into chunks to make it easier to search and retrieve information.

- **Model Selection:** Choose a language model installed locally with Ollama from the list provided. This model will be used to generate the responses to your queries.

<img width="325" alt="Image" src="https://github.com/user-attachments/assets/e10a0e7c-e6b9-4bd3-b3e1-35984d4ad398" />

- **Ask Questions:** Type your questions related to the document. The app will generate accurate, context-aware responses based on the document's content.

- **Retrieval-Augmented Generation:** The system uses RAG to enhance the response by retrieving relevant document excerpts, ensuring your answers are grounded in the document.

## 6. 📦 Project Structure
```bash
project/
├── images/
├── data/
│   └── PDFs/
│   └── sample_question.md
├── notebook/
│   └── main_app.py
│   └── main.ipynb
├── src/
│   └── modules/
│       └── __init__.py
│       └── functions.py
│       └── ingesting.py
│       └── vector_store_module.py
├── README.md
├── requirements.txt
├── requirements.yml
```

## 🛡️ Notes
- Before executing the project, ensure that ollama is running.
- Bear in mind that all processing and inference are performed locally.
- Make sure your system supports running the selected LLM ```(some may require > 8GB RAM).```

## ⚠️ **Limitations**
- **Model Selection:** The accuracy of responses depends on the model you choose. Some models may generate better results than others. It is then a good practice to verify the model that suits your needs.

- **File Size:** Large PDFs may take longer to process and split into chunks. This is because the models are run locally and depends directly on your machine's power.

- **Document Structure:** Although the app is optimised to work with any type of PDFs and docs, complex document structures (e.g., tables, images, non-standard formats) may not always be processed perfectly.

- **Ollama Dependency:** The app relies on the availability of models via Ollama, so ensure the models are properly set up on your machine and running when you test the app.

# 👨‍💻 Author
Made with ❤️ by Alberto AJ - AI/ML Engineer.

## 📢 GitHub Badges


![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/ML-ScikitLearn-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
