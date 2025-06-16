import os
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List
import faiss
import tempfile

# --- ENV VARIABLES ---
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"

# --- PDF LOADER + INDEXING ---
@st.cache_resource(show_spinner=False)
def load_and_index_pdfs(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            docs = loader.load()
            all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(all_splits)
    return vector_store

# --- STATE ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# --- GRAPH ---
def build_graph(vector_store):
    prompt = hub.pull("rlm/rag-prompt")
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")

    def retrieve(state: State):
        docs = vector_store.similarity_search(state["question"])
        return {"context": docs}

    def generate(state: State):
        context_text = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": context_text})
        result = llm.invoke(messages)
        return {"answer": result.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()

# --- APP SETUP ---
st.set_page_config(page_title="📚 PDF Chatbot", layout="wide")
st.title("💬 Multi-PDF Chatbot")
st.markdown("Upload one or more PDF files and ask questions about their content.")

with st.sidebar:
    st.header("📁 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.divider()
    st.header("🕘 Chat History")
    for i, (user, bot) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {user}")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if uploaded_files:
    with st.spinner("📄 Loading PDF(s) and building chatbot..."):
        vector_store = load_and_index_pdfs(uploaded_files)
        graph = build_graph(vector_store)
        
    # --- MAIN CHAT WINDOW ---
    st.subheader("🤖 Ask a question about your PDF(s):")
    user_input = st.chat_input("E.g., 'Summarize the content on page 3'")

    if user_input:
        with st.spinner("🧠 Thinking..."):
            result = graph.invoke({"question": user_input})
            answer = result["answer"]

        st.session_state.chat_history.append((user_input, answer))

    # --- Display Conversation ---
    for user, bot in st.session_state.chat_history[::-1]:
        with st.chat_message("user"):
            st.markdown(user)
        with st.chat_message("assistant"):
            st.markdown(bot)
else:
    st.warning("⬅️ Please upload one or more PDF files to get started.")
