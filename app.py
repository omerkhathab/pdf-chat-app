from utils import generate_response
import os
import shutil
import streamlit as st

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    
if "clear_count" not in st.session_state:
    st.session_state.clear_count = 0
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar:
    st.header("Upload PDFs")
    files = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed", accept_multiple_files=True, key=f"uploader_{st.session_state.clear_count}")
    if files:
        st.session_state.uploaded_files = files
        st.write(f"{len(st.session_state.uploaded_files)} file(s) uploaded.")

    if st.button("Clear Session", type="primary"):
        st.session_state.chat_history.clear()
        st.session_state.uploaded_files = []
        folder_path = "./chroma_store"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        st.session_state.clear_count += 1

st.title("PDF Chat App")
if st.session_state.chat_history:
    for q, (a, srcs) in st.session_state.chat_history:
        with st.chat_message("user", avatar="🔍"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="📚"):
            st.markdown(a)
            with st.expander("Source Chunks"):
                for i, doc in enumerate(srcs):
                    st.markdown(f"**Chunk {i+1}:**", unsafe_allow_html=True)
                    st.text(doc.page_content.strip().replace("\n", " "))
                    # display the metadata
                    if doc.metadata:
                        st.markdown("**Metadata:**")
                        for key, value in doc.metadata.items():
                            st.markdown(f"- `{key}`: {value}")

with st.form("chat_form"):
    col1, col2 = st.columns([5,1])
    query = col1.text_input(
        "ask", 
        key="chat_input_disabled", 
        label_visibility="collapsed", 
        placeholder="Ask something about the uploaded PDFs"
    )
    submitted = col2.form_submit_button("Ask", use_container_width=True)
    if submitted and query:
        if not st.session_state.uploaded_files:
            st.info("Upload one or more PDFs to continue")
        else:
            with st.spinner("Generating answer..."):
                answer, chunks = generate_response(query, st.session_state.uploaded_files)
                st.session_state.chat_history.append((query, (answer, chunks)))
                st.rerun()