import streamlit as st
import rag_service

def render_source_tag(source):
    # Source tag hidden as per user preference
    return

app_title = "DocuMind RAG — Multilingual Document Assistant"
app_description = "Intelligent document QA & summarization supporting English, Hindi, and Marathi with Sarvam AI and Groq."

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
        lang_info = st.session_state["ingested"].get("language", {"code": "en", "name": "English"})
        lang_flag = "🇬🇧" if lang_info.get("code") == "en" else "🇮🇳"
        st.sidebar.write(f"Pages/Chunks processed: {total_pages}")
        st.sidebar.info(f"Detected Language: **{lang_flag} {lang_info.get('name', 'English')}**")
        st.sidebar.success("Document ready for questions.")
else:
    if st.session_state["ingested_file_id"] is not None:
        st.session_state["ingested"] = None
        st.session_state["ingested_file_id"] = None
        st.session_state["chat_history"] = []
    st.sidebar.info("No document uploaded yet.")

# --- MULTILINGUAL SETTINGS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Language Settings")
target_lang_choice = st.sidebar.selectbox(
    "Summary & Response Language",
    options=["Same as Document", "English", "Hindi (हिन्दी)", "Marathi (मराठी)"],
    index=0,
    help="Select the language for summaries and answers (supports cross-lingual generation via Sarvam AI / Groq)."
)
target_lang_map = {
    "Same as Document": "source",
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Marathi (मराठी)": "mr"
}
selected_target_lang = target_lang_map.get(target_lang_choice, "source")

# --- SIDEBAR ACTIONS ---
col1, col2 = st.sidebar.columns(2)
clear_clicked = col1.button("Clear chat")
is_disabled = st.session_state["ingested"] is None
summarize_clicked = col2.button("Full summary", disabled=is_disabled)

if clear_clicked:
    st.session_state["chat_history"] = []
    st.rerun()

if summarize_clicked and st.session_state["ingested"] is not None:
    with st.spinner("Reading document and generating multilingual summary..."):
        try:
            summary, source = rag_service.summarize_document(
                st.session_state["ingested"], 
                target_language=selected_target_lang
            )
            lang_label = target_lang_choice if target_lang_choice != "Same as Document" else "Document's original language"
            st.session_state["chat_history"].append({
                "user": f"Summarize the whole document in {lang_label}.",
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

user_input = st.chat_input("Type your message here (English, Hindi, Marathi)...")

if user_input:
    user_query = user_input.strip()
    with st.chat_message("user"):
        st.write(user_query)
        
    SUMMARY_KEYWORDS = ["summary", "summarize", "summarise", "what is this document about", "what is the document about", "overview of the document", "tl;dr", "सारांश", "संक्षेप", "गोषवारा"]
    
    if st.session_state["ingested"] is None:
        response = "Please upload a document before asking a question."
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state["chat_history"].append({"user": user_query, "bot": response, "source": "System Info"})
    elif any(kw in user_query.lower() for kw in SUMMARY_KEYWORDS):
        with st.chat_message("assistant"):
            with st.spinner("Generating multilingual summary..."):
                try:
                    response, source = rag_service.summarize_document(
                        st.session_state["ingested"], 
                        target_language=selected_target_lang
                    )
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
                    response, source = rag_service.answer_query(
                        st.session_state["ingested"], 
                        user_query, 
                        target_language=selected_target_lang
                    )
                    st.write(response)
                    render_source_tag(source)
                except Exception as e:
                    response = f"Error retrieving answer: {e}"
                    source = "Error"
                    st.write(response)
                st.session_state["chat_history"].append({"user": user_query, "bot": response, "source": source})
                st.rerun()
