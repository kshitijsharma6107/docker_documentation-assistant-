import os
from io import BytesIO
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
from gtts import gTTS

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
os.makedirs(DATA_FOLDER, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

if os.path.exists(DB_FOLDER) and len(os.listdir(DB_FOLDER)) > 0:
    print("loading existing ChromaDB...")
    vectordb = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
else:
    print("no existing database found.")
    print("Building ChromaDB...\n")
    converter = DocumentConverter()
    documents = []
    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)
        try:
            result = converter.convert(path)
            text = result.document.export_to_markdown()
            documents.append(Document(page_content=text, metadata={"source": file}))
            print("Loaded : ", file)
        except Exception as e:
            print("skipped : ", file)
            print(e)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"\nTotal Chunks : {len(chunks)}")
    vectordb = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_FOLDER
    )
    print("\n ChromaDb created successfully! ")

retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_API_KEY)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant specialized in Docker.

Answer ONLY from the given context extracted from Docker-related PDFs.

Context:
{context}

User Question:
{question}

Rules:
- If the answer is not found in the context, reply exactly:
  "I couldn't find that information in the provided documents."
- Answer the question using only the provided context. Be thorough and detailed.
- Mention the source file when possible (e.g., Docker Compose.pdf).
""")

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}" for doc in docs
    )

rag_chain = prompt | llm | StrOutputParser()

# ============ TEXT-TO-SPEECH (gTTS) ============
def speak_text(text: str):
    """Return MP3 bytes for the given text, or None on failure."""
    try:
        tts = gTTS(text=text, lang="en")
        buf = BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return None
# ===============================================

st.set_page_config(page_title="RAG DOCKER", page_icon="💬", layout="wide")
st.title("**DOCKER DOCUMENTATION ASSISTANT 🐳**")
st.write("Ask questions from the documents stored inside the docs folder ")

enable_tts = st.sidebar.checkbox("🔊 Enable Text-to-Speech", value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Replay all previous messages (text + audio + sources) ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("audio"):
            st.audio(message["audio"], format="audio/mp3")
        if message.get("sources"):
            st.markdown("**Sources**")
            for src in message["sources"]:
                st.write(f"- {src}")

question = st.chat_input("Ask a question here... ")

if question:
    # 1. User message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. RAG answer
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = rag_chain.invoke({"context": context, "question": question})

    # 3. Collect sources (deduped)
    sources = []
    if "i couldn't find that information" not in answer.lower():
        for doc in docs:
            src = doc.metadata["source"]
            if src not in sources:
                sources.append(src)

    # 4. Generate audio (only once, stored in session)
    audio_bytes = None
    if enable_tts:
        audio_bytes = speak_text(answer)

    # 5. Persist to session state FIRST
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "audio": audio_bytes,
        "sources": sources,
    })

    # 6. Render current assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        if sources:
            st.markdown("**Sources**")
            for src in sources:
                st.write(f"- {src}")



