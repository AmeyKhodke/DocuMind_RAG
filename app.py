import streamlit as st
import rag_service

def render_source_tag(source):
    if not source:
        return
    
    # Select colors and icons based on the source
    if "groq" in source.lower():
        color = "#00FF66"  # Vibrant neon green
        bg_color = "rgba(0, 255, 102, 0.08)"
        border_color = "rgba(0, 255, 102, 0.25)"
        icon = "⚡"
        badge_text = "LLM Groq API"
    elif "ollama" in source.lower():
        color = "#3399FF"  # Soft sky blue
        bg_color = "rgba(51, 153, 255, 0.08)"
        border_color = "rgba(51, 153, 255, 0.25)"
        icon = "🦙"
        badge_text = source
    elif "heuristic" in source.lower() or "scoring" in source.lower() or "fallback" in source.lower():
        color = "#FF9900"  # Amber/orange
        bg_color = "rgba(255, 153, 0, 0.08)"
        border_color = "rgba(255, 153, 0, 0.25)"
        icon = "⚙️"
        badge_text = source
    else:
        color = "#FF4444"  # Coral red
        bg_color = "rgba(255, 68, 68, 0.08)"
        border_color = "rgba(255, 68, 68, 0.25)"
        icon = "ℹ️"
        badge_text = source

    html = f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        background-color: {bg_color};
        color: {color};
        border: 1px solid {border_color};
        font-weight: 500;
        margin-top: 8px;
        margin-bottom: 2px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: 'Inter', -apple-system, sans-serif;
    ">
        <span>{icon}</span>
        <span>Source: <strong>{badge_text}</strong></span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

app_title = "RAG Application Dashboard"
app_description = "This dashboard allows users to interact with the RAG application through a chat interface."

st.set_page_config(page_title=app_title, page_icon="🤖", layout="wide")
st.title(app_title)
st.markdown(f"### {app_description}")

# --- SESSION STATE INITIALIZATION ---
if "ingested" not in st.session_state:
    st.session_state["ingested"] = None
if "ingested_file_id" not in st.session_state:
    st.session_state["ingested_file_id"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Document Upload")
st.sidebar.write("Upload a PDF, DOCX, or TXT file to ask questions about it.")

file = st.sidebar.file_uploader("Upload your documents here", type=["pdf", "docx", "txt"], key="file_uploader")

if file is not None:
    st.sidebar.success(f"Selected file: {file.name}")
    st.sidebar.write(f"File size: {round(file.size / 1024 / 1024, 2)} MB")
    file_id = f"{file.name}-{file.size}"
    
    # Process only if it is a completely new file
    if st.session_state["ingested_file_id"] != file_id:
        with st.spinner("Processing document (chunking + embeddings)..."):
            try:
                st.session_state["ingested"] = rag_service.ingest_document(file)
                st.session_state["ingested_file_id"] = file_id
                st.session_state["chat_history"] = []  # Clear old history for a fresh document
            except Exception as e:
                st.sidebar.error(f"Failed to process document: {e}")
                st.session_state["ingested"] = None
                st.session_state["ingested_file_id"] = None

    if st.session_state["ingested"] is not None:
        total_pages = st.session_state["ingested"].get("total_pages", "N/A")
        st.sidebar.write(f"Pages/Chunks processed: {total_pages}")
        st.sidebar.success("Document ready for questions.")
else:
    if st.session_state["ingested_file_id"] is not None:
        st.session_state["ingested"] = None
        st.session_state["ingested_file_id"] = None
        st.session_state["chat_history"] = []
    st.sidebar.info("No document uploaded yet.")

# --- SIDEBAR ACTIONS ---
col1, col2 = st.sidebar.columns(2)
clear_clicked = col1.button("Clear chat")
is_disabled = st.session_state["ingested"] is None
summarize_clicked = col2.button("Full summary", disabled=is_disabled)

if clear_clicked:
    st.session_state["chat_history"] = []
    st.rerun()

if summarize_clicked and st.session_state["ingested"] is not None:
    with st.spinner("Reading document and summarizing..."):
        try:
            summary, source = rag_service.summarize_document(st.session_state["ingested"])
            st.session_state["chat_history"].append({
                "user": "Summarize the whole document.",
                "bot": summary,
                "source": source
            })
            st.rerun()
        except Exception as e:
            st.error(f"Summarization failed: {e}")

# --- CHAT INTERFACE RENDERING ---
for chat in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant"):
        st.write(chat["bot"])
        if "source" in chat and chat["source"]:
            render_source_tag(chat["source"])

user_input = st.chat_input("Type your message here...")

if user_input:
    user_query = user_input.strip()
    with st.chat_message("user"):
        st.write(user_query)
        
    SUMMARY_KEYWORDS = ["summary", "summarize", "summarise", "what is this document about", "what is the document about", "overview of the document", "tl;dr"]
    
    if st.session_state["ingested"] is None:
        response = "Please upload a document before asking a question."
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state["chat_history"].append({"user": user_query, "bot": response, "source": "System Info"})
    elif any(kw in user_query.lower() for kw in SUMMARY_KEYWORDS):
        with st.chat_message("assistant"):
            with st.spinner("Reading the whole document to summarize it (this may take a bit)..."):
                try:
                    response, source = rag_service.summarize_document(st.session_state["ingested"])
                    st.write(response)
                    render_source_tag(source)
                except Exception as e:
                    response = f"Error during summary extraction: {e}"
                    source = "Error"
                    st.write(response)
                st.session_state["chat_history"].append({"user": user_query, "bot": response, "source": source})
                st.rerun()
    else:
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    response, source = rag_service.answer_query(st.session_state["ingested"], user_query)
                    st.write(response)
                    render_source_tag(source)
                except Exception as e:
                    response = f"Error retrieving answer: {e}"
                    source = "Error"
                    st.write(response)
                st.session_state["chat_history"].append({"user": user_query, "bot": response, "source": source})
                st.rerun()
