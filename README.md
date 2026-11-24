# 🐳 Docker Documentation Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to ask questions about Docker documentation and receive accurate, context-aware answers from uploaded documents.

Built using **Python, Streamlit, LangChain, ChromaDB, Hugging Face Embeddings, Docling and Groq LLM**.

## 🚀 Features

- **Document Upload:** Upload PDF, DOCX and PPTX files.
- **Document Processing:** Extracts document content using Docling.
- **Text Chunking:** Splits extracted text into smaller chunks using RecursiveCharacterTextSplitter.
- **Semantic Search:** Retrieves relevant document chunks using vector similarity search.
- **Vector Database:** Stores embeddings persistently using ChromaDB.
- **Incremental Indexing:** Processes only newly uploaded documents instead of re-embedding existing documents.
- **AI-Powered Answers:** Uses Groq LLM to generate answers based on retrieved document context.
- **Source Attribution:** Displays the source documents used to generate answers.
- **GPU Acceleration:** Supports CUDA for faster embedding generation when a compatible GPU is available.
- **Chat Interface:** Provides an interactive chat interface using Streamlit.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | RAG pipeline orchestration |
| Docling | Document conversion and text extraction |
| Hugging Face | Text embedding generation |
| BGE Base Embeddings | Semantic text embeddings |
| ChromaDB | Persistent vector database |
| Groq | LLM inference |
| PyTorch | GPU acceleration using CUDA |

## ⚙️ How It Works

The application follows a Retrieval-Augmented Generation (RAG) architecture.

1. **Document Upload:** Users upload PDF, DOCX or PPTX documents.
2. **Document Conversion:** Docling extracts document content and converts it into Markdown.
3. **Text Chunking:** The extracted content is divided into smaller chunks.
4. **Embedding Generation:** Hugging Face BGE embeddings convert text chunks into numerical vectors.
5. **Vector Storage:** ChromaDB stores the embeddings along with document metadata.
6. **Retrieval:** When a user asks a question, the retriever searches for the most relevant chunks.
7. **Response Generation:** Groq LLM generates an answer using the retrieved context.
8. **Source Display:** The application displays the source documents used for the answer.

## 🏗️ Project Architecture

```text
          User
           |
           v
    Streamlit Interface
           |
           v
     Document Upload
           |
           v
     Docling Converter
           |
           v
     Text Extraction
           |
           v
     Text Chunking
           |
           v
   Hugging Face Embeddings
           |
           v
       ChromaDB
           |
           |
User Question
      |
      v
  Retriever
      |
      v
Relevant Document Chunks
      |
      v
   Prompt Template
      |
      v
     Groq LLM
      |
      v
  Generated Answer
      |
      v
  Streamlit Chat UI
```

## 📂 Project Structure

```text
docker_documentation-assistant/
│
├── app.py
├── docs/
│   └── (Uploaded documents)
│
├── chromadb/
│   └── (Persistent vector database)
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kshitijsharma6107/docker_documentation-assistant-.git
```

### 2. Navigate to the project directory

```bash
cd docker_documentation-assistant-
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure environment variables

Create a `.env` file in the project root directory.

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from [Groq Console](https://console.groq.com/).

### 7. Run the application

```bash
streamlit run app.py
```

## 🧠 Models Used

**Embedding Model:**
- `BAAI/bge-base-en-v1.5`
- Used to generate semantic embeddings for document chunks.

**Large Language Model:**
- `openai/gpt-oss-20b` through Groq.
- Used to generate context-aware responses.

## ⚡ GPU Acceleration

The application uses PyTorch to detect CUDA availability.

- If CUDA is available, the embedding model runs on the GPU.
- Otherwise, it automatically falls back to the CPU.

## 🔒 Security

- API keys are stored in environment variables using `.env`.
- Sensitive files and local virtual environments should not be committed to GitHub.
- The `.env` file is excluded using `.gitignore`.

## 🔮 Future Improvements

- Voice input for asking questions.
- Image and diagram understanding in PDFs.
- Improved document update and deletion detection.
- Support for additional document formats.
- Chat history management.
- Better document retrieval and reranking.

## 👨‍💻 Author

**Kshitij Sharma**

Engineering Student | AI/ML Enthusiast

GitHub: [@kshitijsharma6107](https://github.com/kshitijsharma6107)

---

⭐ If you find this project interesting, consider giving it a star!
