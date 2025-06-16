# 💬 Multi-PDF Chatbot

This project is a **Streamlit-based chatbot** that allows users to upload one or more PDF files and ask natural language questions about their content. The system uses **LangChain**, **Hugging Face Embeddings**, **FAISS**, and **LLM from Groq** (LLaMA3-8B) to provide accurate responses based on the PDF documents.

---

## 🚀 Features

- Upload multiple PDFs and query their contents.
- Uses Sentence Transformers for embeddings.
- FAISS for vector similarity search.
- Groq-hosted LLaMA 3 model for high-performance LLM responses.
- Memory-efficient: Embedding and indexing done in cached Streamlit function.
- Chat history panel and clear button for session management.

---

## 📦 Installation

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🌐 Environment Variables

Set the following environment variables before running the app:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_langsmith_api_key
export GROQ_API_KEY=your_groq_api_key
```

### 🔑 Get Your API Keys:

- **LangSmith API Key**: https://docs.smith.langchain.com/user_guide/welcome  
- **Groq API Key (for LLaMA 3)**: https://console.groq.com/keys  

> Replace `your_langsmith_api_key` and `your_groq_api_key` with your actual keys.

---

## 📂 File Structure

```
.
├── app.py              # Main Streamlit app file
├── requirements.txt    # Python dependencies
├── README.md           # This file
```

---

## 🧠 How It Works

1. **PDF Upload**: Upload one or more PDFs via the sidebar.
2. **Text Extraction**: PDFs are loaded and split into chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding + Indexing**: Text chunks are converted into vector embeddings using `sentence-transformers/all-mpnet-base-v2`, and stored in a FAISS index.
4. **Chat Interface**: Ask questions in the chat box. The system performs similarity search and passes the results to LLaMA 3 hosted on Groq to generate a response.
5. **Session History**: View and clear your chat history.

---

## ▶️ Running the App

After setting up your environment and API keys, run:

```bash
streamlit run app.py
```

Then open the link shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## 📘 Example Prompts

- "Summarize the content on page 3."
- "What are the key findings in the conclusion?"
- "Explain the methodology used in the second document."

---

## 🧹 Clear Chat History

Click the **🧹 Clear Chat** button in the sidebar to reset the conversation history.

---
