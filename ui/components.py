import streamlit as st
from typing import List
import tempfile
import os


def retrieval_mode_selector() -> str:
    """
    Allows the user to select the retrieval mode for answering queries.

    Args:
        None

    Returns:
        The selected retrieval mode as a string.
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
    """
    Displays a button to trigger document processing.

    Args:
        None

    Returns:
        True if the button is clicked, otherwise False.
    """
    return st.button(
        "🚀 Process & Index Documents",
        help="Click to chunk and embed uploaded documents"
    )


def display_answer_metadata():
    """
    Displays metadata related to the most recent answer.

    Args:
        None

    Returns:
        None
    """
    meta = st.session_state.get("last_answer_meta")
    if not meta:
        return

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

    if meta.get("doc_summaries"):
        with st.expander("📝 Top Document Summaries", expanded=False):
            for summary in meta["doc_summaries"]:
                st.markdown(f"- {summary}")


def init_session_state():
    """
    Initializes required Streamlit session state variables.

    Args:
        None

    Returns:
        None
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()


def display_chat_history():
    """
    Displays the full chat history.

    Args:
        None

    Returns:
        None
    """
    init_session_state()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")


def add_message(role: str, content: str, sources: List[str] = None):
    """
    Adds a message to the chat history.

    Args:
        role: Role of the message sender.
        content: Message content.
        sources: Optional list of sources.

    Returns:
        None
    """
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
    st.session_state.messages.append(message)


def clear_chat_history():
    """
    Clears all chat messages.

    Args:
        None

    Returns:
        None
    """
    st.session_state.messages = []


def save_uploaded_file(uploaded_file) -> str:
    """
    Saves an uploaded file to a temporary directory.

    Args:
        uploaded_file: File uploaded through Streamlit.

    Returns:
        Path to the saved file.
    """
    init_session_state()

    file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def display_sidebar_info():
    """
    Displays sidebar content and controls.

    Args:
        None

    Returns:
        None
    """
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


def display_file_uploader():
    """
    Displays the document upload widget.

    Args:
        None

    Returns:
        Uploaded files.
    """
    return st.file_uploader(
        "Upload your documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents to chat with"
    )


def display_processing_status(message: str, status: str = "info"):
    """
    Displays a status message in the UI.

    Args:
        message: Message to display.
        status: Type of status message.

    Returns:
        None
    """
    if status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(message)


def create_web_search_toggle() -> bool:
    """
    Displays a toggle to enable or disable web search.

    Args:
        None

    Returns:
        True if enabled, otherwise False.
    """
    return st.toggle(
        "🌐 Enable Web Search",
        value=False,
        help="When enabled, the chatbot will also search the web"
    )


def display_evidence(answer: str, docs: list, web_used: bool):
    """
    Displays answer content along with document and web evidence.

    Args:
        answer: Generated answer text.
        docs: List of document evidence.
        web_used: Flag indicating whether web evidence was used.

    Returns:
        None
    """
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
