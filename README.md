# DocuMind_RAG — Latency-Optimized Multilingual RAG Application

A lightweight, high-performance Retrieval-Augmented Generation (RAG) system built from scratch in Python and Streamlit without third-party RAG orchestration frameworks (no LangChain, no LlamaIndex).

---

## Features
- **Hand-rolled RAG Architecture**: Complete control over document loading (`pypdf`, `python-docx`, `txt`), custom chunking, `sentence-transformers` embeddings, and `ChromaDB` vector storage.
- **Latency Optimized**:
  - `@st.cache_resource` singleton caching for embedding models and ChromaDB clients.
  - On-disk hash caching (`.cache/`) to avoid re-chunking and re-embedding previously processed documents.
  - Parallelized PDF text extraction using thread pools.
  - In-memory query caching (LRU/dictionary).
  - Real-time token streaming for English answers using Groq SSE & `st.write_stream`.
  - Optimized Groq model priority list with fallback to local Ollama (`gemma:2b`) and internal heuristic scoring.
- **Multilingual Support (English / Hindi / Marathi)**:
  - Select answer language from sidebar.
  - **3-Tier Resilient Translation Fallback**:
    1. **Tier 1 (Primary)**: Dedicated Indic translation via **Sarvam AI Translate REST API** (`https://api.sarvam.ai/translate`).
    2. **Tier 2 (Fallback)**: LLM translation pass via Groq (`llama-3.1-8b-instant`) or local Ollama.
    3. **Tier 3 (Last Resort)**: Return original English text with an informative note (`Translation Unavailable`).

---

## Quickstart

### 1. Installation
Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/AmeyKhodke/DocuMind_RAG.git
cd DocuMind_RAG
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Setup
Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and insert your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
HF_TOKEN=your_hf_token_here
```

#### How to get a free Sarvam AI API Key:
1. Visit [Sarvam AI Dashboard](https://dashboard.sarvam.ai/).
2. Sign up / log in to your account.
3. Navigate to **API Keys** and generate a new key.
4. Copy the key and paste it as `SARVAM_API_KEY` in your `.env` file or Streamlit secrets.

### 3. Run the Dashboard

```bash
streamlit run app.py
```

---

## How to Add New Languages (Extensibility)

Adding support for additional Indic or global languages takes just two simple updates:

1. **Update `rag_service.py`**: Add the language name and its corresponding BCP-47 code to `LANGUAGE_CODES`:

```python
LANGUAGE_CODES = {
    "English": "en-IN",
    "हिंदी (Hindi)": "hi-IN",
    "मराठी (Marathi)": "mr-IN",
    "ગુજરાતી (Gujarati)": "gu-IN",   # Example 4th language
    "தமிழ் (Tamil)": "ta-IN"          # Example 5th language
}
```

2. **No UI changes required**: `app.py` automatically populates the language selection dropdown directly from `rag_service.LANGUAGE_CODES.keys()`.

---

## Testing & Profiling

Run the standalone profiling and fallback test suite:

```bash
python test_and_profile.py
```

For detailed performance numbers and architectural insights, refer to [PERFORMANCE.md](PERFORMANCE.md).
