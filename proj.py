
import os
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import streamlit as st
import torch

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY not found inside .env ")

DATA_FOLDER = "docs"
DB_FOLDER = "chromadb"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_NAME = "openai/gpt-oss-20b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
TOP_K = 9

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".pptx")

os.makedirs(DATA_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 8
    }
)



splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

if os.path.exists(DB_FOLDER) and len(os.listdir(DB_FOLDER)) > 0:

    print("loading existing ChromaDB...")

    vectordb = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embeddings
    )

else:

    print("no existing database found.")
    print("Building ChromaDB...\n")

    converter = DocumentConverter()
    documents = []

    for file in os.listdir(DATA_FOLDER):

        if not file.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        path = os.path.join(DATA_FOLDER, file)

        try:

            result = converter.convert(path)
            text = result.document.export_to_markdown()

            documents.append(
                Document(page_content=text, metadata={"source": file})
            )

            print("Loaded : ", file)

        except Exception as e:

            print("skipped : ", file)
            print(e)

    chunks = splitter.split_documents(documents)

    print(f"\nTotal Chunks : {len(chunks)}")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_FOLDER
    )

    print("\n ChromaDb created successfully! ")


# NEW: Incremental indexing function
# Checks existing filenames and indexes only new documents.

def index_new_documents():

    converter = DocumentConverter()

    existing = vectordb.get(include=["metadatas"])

    existing_files = set()

    for doc in existing["metadatas"]:

       if doc and "source" in doc:
          filename = doc["source"]
          existing_files.add(filename)
    for file in os.listdir(DATA_FOLDER):

        if not file.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        if file in existing_files:
            continue

        try:

            path = os.path.join(DATA_FOLDER, file)

            result = converter.convert(path)
            text = result.document.export_to_markdown()

            doc = Document(
                page_content=text,
                metadata={"source": file}
            )

            chunks = splitter.split_documents([doc])

            if chunks:
                vectordb.add_documents(chunks)

            print("Indexed new document:", file)

        except Exception as e:

            print("Error indexing:", file)
            print(e)


# NEW: Automatically check for new documents
index_new_documents()


#..................................retrieval..............................

retriever = vectordb.as_retriever(
    search_kwargs={"k": TOP_K}
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant specialized in Docker.

Answer ONLY from the given context extracted from Docker-related documents.

Context:
{context}

User Question:
{question}

Rules:
- If the answer is not found in the context, reply exactly:
  "I couldn't find that information in the provided documents."
- Keep answers concise, clear, and structured.
- Use only the provided context.
- Include relevant commands and steps when necessary.
- Mention the source file whenever possible.
""")

def format_docs(docs):

    return "\n\n".join(
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
        for doc in docs
    )

rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

st.set_page_config(
    page_title="RAG DOCKER",
    page_icon="💬",
    layout="wide"
)

st.title("**DOCKER DOCUMENTATION ASSISTANT 🐳**")

st.write("Ask questions from the documents stored inside the docs folder ")

# -------------  UPLOAD NEW DOCUMENTS 

uploaded_files = st.file_uploader(
    "📎 Upload PDF / DOCX / PPTX",
    type=["pdf", "docx", "pptx"],
    accept_multiple_files=True
)

if st.button("🚀 Upload and Index Documents"):

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_path = os.path.join(
                DATA_FOLDER,
                uploaded_file.name
            )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())    #### uploaded file is the file that we uploaded and we save iits content in new file f 

        index_new_documents()

        st.success("Documents uploaded and indexed successfully!")

    else:
        st.warning("Please upload a document first.")


if "messages" not in st.session_state:

    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

question = st.chat_input("Ask a question here... ")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    docs = retriever.invoke(question)

    context = format_docs(docs)

    answer = rag_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)

        if "i couldn't find that information" not in answer.lower():

            st.markdown("**Sources**")

            shown = []

            for doc in docs:

                source = doc.metadata["source"]

                if source not in shown:

                    st.write(f"-{source}")

                    shown.append(source)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Keep answers concise, clear, and structured
