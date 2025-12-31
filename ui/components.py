import streamlit as st
from typing import List
import tempfile
import os


# --------------------------------------------------
# Session State
# --------------------------------------------------

def retrieval_mode_selector() -> str:
    """
    Let user choose retrieval strategy.
    """
    return st.radio(
        "🔍 Retrieval Mode",
        options=["doc", "web", "hybrid"],
        format_func=lambda x: {
            "doc": "📄 Document-based",
            "web": "🌐 Web-based",
            "hybrid": "🔀 Hybrid"
        }[x],
        index=0
    )
def process_documents_button() -> bool:
    return st.button(
        "🚀 Process & Index Documents",
        help="Click to chunk and embed uploaded documents"
    )


import streamlit as st


def display_answer_metadata():
    meta = st.session_state.get("last_answer_meta")
    if not meta:
        return

    # ---------- Visual Indicator (smaller heading) ----------
    indicator_map = {
        "doc": "📄 Document-based answer",
        "web": "🌐 Web-based answer",
        "hybrid": "🔀 Hybrid answer"
    }

    st.markdown(
        f"<p style='font-size:16px; font-weight:600;'>"
        f"{indicator_map.get(meta['answer_type'], '')}"
        f"</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ---------- Citations (Dropdown) ----------
    with st.expander("📌 Citations", expanded=False):
        if not meta.get("doc_chunks") and not meta.get("web_docs"):
            st.write("No citations available.")
        else:
            for i, doc in enumerate(meta.get("doc_chunks", []), 1):
                st.markdown(
                    f"- **[Doc]** {doc.metadata.get('source')} – Chunk{i}"
                )

            for web in meta.get("web_docs", []):
                st.markdown(
                    f"- **[Web]** Tavily: “{web.metadata.get('title', 'Unknown')}”"
                )

    # ---------- Top-N Summaries (Dropdown) ----------
    if meta.get("doc_summaries"):
        with st.expander("📝 Top Document Summaries", expanded=False):
            for summary in meta["doc_summaries"]:
                st.markdown(f"- {summary}")


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()


# --------------------------------------------------
# Chat History
# --------------------------------------------------
def display_chat_history():
    """Display all messages in the chat history."""
    init_session_state()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")


def add_message(role: str, content: str, sources: List[str] = None):
    """Add a message to chat history."""
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
    st.session_state.messages.append(message)


def clear_chat_history():
    """Clear all messages from chat history."""
    st.session_state.messages = []


# --------------------------------------------------
# File Handling
# --------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file to a temporary session directory.
    """
    init_session_state()

    file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
def display_sidebar_info():
    """Display information in the sidebar."""
    init_session_state()

    with st.sidebar:
        st.header("📖 About")
        st.markdown("""
        This is a **Hybrid RAG Chatbot** that can:
        - 📄 Answer questions from your documents
        - 🌐 Search the web using Tavily
        - 💬 Provide grounded, citation-aware responses

        **How to use:**
        1. Upload PDF or TXT files
        2. Wait for indexing to complete
        3. Ask your questions
        """)

        st.divider()

        st.header("📁 Uploaded Files")
        if st.session_state.uploaded_files:
            for file in st.session_state.uploaded_files:
                st.write(f"✅ {file}")
        else:
            st.write("No files uploaded yet")

        st.divider()

        if st.button("🗑️ Clear Chat History"):
            clear_chat_history()
            st.rerun()


# --------------------------------------------------
# Widgets
# --------------------------------------------------
def display_file_uploader():
    """Display file upload widget."""
    return st.file_uploader(
        "Upload your documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents to chat with"
    )


def display_processing_status(message: str, status: str = "info"):
    """Display a status message."""
    if status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(message)


def create_web_search_toggle() -> bool:
    """Create a toggle for enabling web search."""
    return st.toggle(
        "🌐 Enable Web Search",
        value=False,
        help="When enabled, the chatbot will also search the web"
    )

def display_evidence(answer: str, docs: list, web_used: bool):
    tabs = st.tabs(["✅ Answer", "📄 Documents", "🌐 Web"])

    with tabs[0]:
        st.markdown(answer)

    with tabs[1]:
        for d in docs:
            st.markdown(f"**{d.metadata.get('source')}**")
            st.write(d.page_content[:500])

    with tabs[2]:
        if web_used:
            st.info("Web evidence used")
        else:
            st.write("No web evidence")
