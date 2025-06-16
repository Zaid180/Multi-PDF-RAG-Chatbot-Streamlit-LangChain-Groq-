# 💬 Multi-PDF Chatbot using RAG (LangChain + HuggingFace + FAISS + Groq)

This is a **Streamlit web application** that allows you to **upload multiple PDF documents** and ask natural language questions about their content. It uses a **RAG (Retrieval-Augmented Generation)** pipeline to answer questions accurately using contextual information from the uploaded PDFs.

---

## 🚀 Features

- 📄 Upload and chat with **multiple PDFs**
- 🧠 Built using **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Groq's LLaMA3**
- 💾 Uses in-memory FAISS vector store for fast retrieval
- 🧾 Automatically splits, indexes, and stores your documents
- 🤖 Chat-like interface using **Streamlit's new chat UI**
- 🕘 Sidebar for persistent **chat history**
- 🔄 Option to **clear history** anytime

---

## 🧰 Requirements

Install all necessary dependencies in one go:

```bash
pip install -r requirements.txt
requirements.txt
txt
Copy
Edit
streamlit
langchain
langchain-community
langchain-core
langchainhub
langchain-huggingface
sentence-transformers
faiss-cpu
PyPDF2
typing-extensions
💡 You can also use pip install faiss-gpu if you have a GPU-enabled setup.

🔑 Environment Variables
Set these environment variables in your terminal or .env file:

bash
Copy
Edit
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=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
export GROQ_API_KEY=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
🛡️ Make sure to keep these API keys secret in production.

📂 File Structure
bash
Copy
Edit
📦 multi-pdf-chatbot/
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── README.md             # This file
🧪 How to Run Locally
Clone this repository:

bash
Copy
Edit
git clone https://github.com/your-username/multi-pdf-chatbot.git
cd multi-pdf-chatbot
Install Python packages:

bash
Copy
Edit
pip install -r requirements.txt
Set the required environment variables:

bash
Copy
Edit
export LANGSMITH_API_KEY=your_langsmith_key
export GROQ_API_KEY=your_groq_key
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Open the app in your browser at http://localhost:8501

📦 Models & Tools Used
LLaMA3 (8B)

HuggingFace Embeddings

FAISS Vector Store

LangChain Framework

Streamlit

✨ Demo Features
Upload PDFs directly from the sidebar

Automatically chunk and embed content

Use similarity search with FAISS

Generate answers using Groq's fast LLaMA model

Clean UI with persistent chat history

⚠️ Notes
Do not share your API keys publicly.

This app uses in-memory FAISS, so it doesn't persist across sessions.

Only PDFs are supported at this time.

📧 Contact
For issues or collaboration ideas, feel free to open an Issue or reach out via LinkedIn.

Made with ❤️ using LangChain, Streamlit, and Groq.

yaml
Copy
Edit

---

Let me know if you'd like the README in a different format (e.g., with badges, for GitHub Pages, etc.).
