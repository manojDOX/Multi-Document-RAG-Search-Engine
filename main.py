import streamlit as st

from ui.chat import ChatInterface
from ui.components import (
    init_session_state,
    display_chat_history,
    display_sidebar_info,
    display_file_uploader,
    add_message,
    display_answer_metadata,
    retrieval_mode_selector,
    process_documents_button,
)

st.set_page_config(
    page_title="Hybrid RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


def main():
    # ---------------- Session Init ----------------
    init_session_state()
    display_sidebar_info()

    if "chat_interface" not in st.session_state:
        st.session_state.chat_interface = ChatInterface()

    chat_interface: ChatInterface = st.session_state.chat_interface

    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        retrieval_mode = retrieval_mode_selector()
        uploaded_files = display_file_uploader()
        process_clicked = process_documents_button()

    # ---------------- Document Processing ----------------
    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing and indexing documents..."):
                chunk_count = chat_interface.process_uploaded_files(uploaded_files)
                st.success(f"✅ Indexed {chunk_count} chunks")

    # ---------------- Chat History ----------------
    display_chat_history()

    # ---------------- Chat Input ----------------
    user_query = st.chat_input("Ask a question...")

    if user_query:
        add_message("user", user_query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            for chunk in chat_interface.get_response(
                query=user_query,
                retrieval_mode=retrieval_mode
            ):
                full_response += chunk
                placeholder.markdown(full_response)

        add_message(
            "assistant",
            full_response,
            chat_interface.get_sources(user_query, retrieval_mode)
        )

        # ---- Visual indicators + citations + summaries ----
        display_answer_metadata()


if __name__ == "__main__":
    main()
