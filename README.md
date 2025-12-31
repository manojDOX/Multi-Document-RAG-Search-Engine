

# 🧠 Multi-Document Hybrid RAG Search Engine (GA02)

Name: Manoj B

Mail Id: manoj2882003@gmail.com

**[APP LINK](https://multi-document-rag-search-engine-dox.streamlit.app/)**

A **Hybrid Retrieval-Augmented Generation (RAG) Search Engine** that allows users to query **multiple uploaded documents** and optionally augment answers with **real-time web search** using Tavily — all through a clean **Streamlit chatbot UI**.

This project mirrors real-world **enterprise copilots and research assistants** that combine **private knowledge bases** with **live internet data**, while maintaining transparency through **citations and summaries**.

---

## 🚀 Project Purpose

Organizations store knowledge across **unstructured documents** such as PDFs and text files.
However, static document knowledge alone is often insufficient — users may also require **up-to-date, real-world information**.

This project addresses that gap by providing:

* 🔍 **Semantic search across multiple documents**
* 📄 **Document-grounded question answering**
* 🌐 **Optional real-time web search (Tavily)**
* 🔀 **Hybrid answers combining documents + web**
* 📌 **Clear citations & evidence**
* 📝 **Top-N document summaries**
* 🤖 **Interactive Streamlit chatbot interface**

---

## ✨ Key Features

* **Multi-document ingestion** (PDF, TXT)
* **FAISS vector database** for fast semantic retrieval
* **HuggingFace sentence-transformer embeddings** (local & free)
* **Groq LLM integration** for fast inference
* **Tavily web search** for real-time information
* **User-selectable retrieval mode**:

  * 📄 Document-based
  * 🌐 Web-based
  * 🔀 Hybrid
* **Explicit document indexing button** (no accidental re-embedding)
* **Citation-aware answers**
* **Dropdown-based evidence & summaries**

---

## 🧱 Tech Stack

| Layer             | Technology                            |
| ----------------- | ------------------------------------- |
| Language          | Python                                |
| LLM Orchestration | LangChain                             |
| LLM Provider      | Groq                                  |
| Embeddings        | HuggingFace (`sentence-transformers`) |
| Vector Store      | FAISS                                 |
| Web Search        | Tavily                                |
| Frontend          | Streamlit                             |

---

## 📁 Project Structure

```
Multi-Document-RAG-Search-Engine/
├── config/
│   └── settings.py
├── core/
│   ├── ingestion.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── chain.py
├── tools/
│   └── tavily_search.py
├── ui/
│   ├── chat.py
│   └── components.py
├── data/
│   └── faiss_index/
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Multi-Document-RAG-Search-Engine.git
cd Multi-Document-RAG-Search-Engine
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

⚠️ **Important**: Ensure `sentence-transformers` is installed
(it is required for embeddings).

---

### 4️⃣ Set Environment Variables

Create a `.env` file in the root directory:

```env
GPT_MODEL_NAME=
GROQ_API_KEY=
TEMPRATURE=
FAISS_INDEX_PATH=
EMBEDDING_MODEL=
CHUNK_SIZE=
CHUNK_OVERLAP=
TOP_K_RESULTS=
TAVILY_API_KEY=
TOP_K_WEB_RESULTS=
```

---

### 5️⃣ Run the Application

```bash
streamlit run main.py
```

The app will open in your browser 🚀

---

## 🧑‍💻 How to Use

1. **Upload documents** (PDF or TXT)
2. Click **🚀 Process & Index Documents**
3. Choose retrieval mode:

   * 📄 Document-based
   * 🌐 Web-based
   * 🔀 Hybrid
4. Ask questions in the chat
5. View:

   * Answer
   * 📌 Citations (dropdown)
   * 📝 Top-N document summaries (dropdown)

---

## 📘 Documentation

### 🏗 Architecture Diagram (Logical Flow)

```
User Query
   │
   ▼
Streamlit UI
   │
   ├── Document Retrieval (FAISS)
   │       └── HuggingFace Embeddings
   │
   ├── Web Retrieval (Tavily)
   │
   ▼
Context Assembly (Doc / Web / Hybrid)
   │
   ▼
Groq LLM
   │
   ▼
Answer + Citations + Summaries
```

---

### 🧠 Design Rationale

* **Explicit user control** over retrieval mode improves transparency
* **FAISS + local embeddings** avoid external vector DB costs
* **Session-state persistence** prevents re-embedding on reruns
* **Expanders for citations/summaries** reduce UI clutter
* **Hybrid RAG** reflects real enterprise knowledge workflows

---

### ❓ Example Queries

**Document-based**

* “Explain Infosys trainee hiring process”
* “What skills are required for freshers?”

**Web-based**

* “Latest Infosys hiring news 2026”
* “Current IT hiring trends in India”

**Hybrid**

* “Compare Infosys hiring plans with current IT market trends”
* “How do Infosys internal plans align with recent industry news?”

---

### 📊 Evaluation Report (Summary)

| Aspect              | Observation                           |
| ------------------- | ------------------------------------- |
| Retrieval relevance | High for document-grounded queries    |
| Answer grounding    | Strong, with explicit citations       |
| Hybrid reasoning    | Effective when both sources available |
| Latency             | Low (Groq + FAISS)                    |
| Transparency        | High (citations + summaries)          |

**Limitations**

* No reranking model
* Web results depend on Tavily quality

**Future Enhancements**

* Answer confidence scoring
* RAG evaluation metrics
* Evidence highlighting
* Multi-user persistence

---

## 📌 License

This project is for **educational and portfolio purposes**.
You are free to modify and extend it.

---



