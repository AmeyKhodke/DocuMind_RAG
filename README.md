# 🤖 DocuMind RAG — Multilingual Document Assistant

**DocuMind RAG** is an intelligent, high-speed Retrieval-Augmented Generation (RAG) platform supporting document question answering and cross-lingual summarization across **English, Hindi, and Marathi**.

Powered by **Sarvam AI** (specialized Indic models) and **Groq** (ultra-fast LLM inference), DocuMind RAG understands native Devanagari script, filters language-specific stopwords, applies language-aware extractive compression, and delivers fluent cross-lingual responses.

---

## 🌟 Key Features

* **🌐 Native Multilingual Support**: Seamless understanding of documents in **English**, **Hindi (हिन्दी)**, and **Marathi (मराठी)**.
* **🔍 Automatic Language Detection**: Zero-API script analysis and lexical classification detecting English, Hindi, and Marathi in microseconds.
* **🇮🇳 Sarvam AI Integration**: Direct integration with Sarvam AI's Indic models (sarvam-105b) for culturally nuanced and accurate Indian language processing.
* **⚡ Groq LLM Acceleration**: Lightning-fast fallback with open-weights LLMs (openai/gpt-oss-120b, llama-3.3-70b-versatile, qwen/qwen3.8-27b).
* **🔄 Cross-Lingual Summarization & QA**:
  * Summarize an English document in Marathi or Hindi.
  * Ask questions in Hindi about an English or Marathi document.
  * Target output language selector (Same as Document, English, Hindi, Marathi).
* **📑 Devanagari-Aware Sentence Segmentation**: Correctly splits sentences using both standard punctuation (., !, ?) and Devanagari dandas (।, ॥).
* **🛑 Curated Indic Stopword Filtering**: Dedicated stopword dictionaries for Hindi and Marathi preventing grammatical particles from dominating keyword ranking.
* **📦 Lightweight Extractive Compression**: Pre-compresses large documents to preserve context and reduce API costs while preserving document flow.
* **💾 ChromaDB Vector Database**: Local, persistent vector store with semantic embeddings via SentenceTransformer.
* **🎨 Interactive Streamlit UI**: User-friendly chat dashboard with file upload, document statistics, and language preferences.

---

## 🏗️ Architecture

`
[ Upload Document (PDF / DOCX / TXT) ]
                  │
                  ▼
   [ Language Detection (en / hi / mr) ]
                  │
                  ▼
   [ Chunking (Danda '।', '॥' & '.' Aware) ]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
[ ChromaDB Embeddings ]   [ Indic Stopword Filtering & Sentence Ranking ]
        │                                   │
        ▼                                   ▼
 [ Vector Search ]                  [ Context Compression ]
        └─────────┬─────────────────────────┘
                  │
                  ▼
         [ LLM Router ]
        ┌─────────┴─────────┐
        ▼                   ▼
[ Sarvam AI API ]     [ Groq LLM API ]
 (Indic Primary)     (Multilingual Fast Fallback)
        │                   │
        └─────────┬─────────┘
                  ▼
[ Cross-Lingual Summary / QA Output ]
`

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
`ash
git clone https://github.com/AmeyKhodke/DocuMind_RAG.git
cd DocuMind_RAG
`

### 2. Create and Activate Virtual Environment
`ash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
`

### 3. Install Dependencies
`ash
pip install -r requirements.txt
`

### 4. Configure Environment Variables
Copy .env.example to .env:
`ash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
`

Open .env and add your API keys:
`env
# Sarvam AI API Key (required for Indic models)
# Get yours from: https://www.sarvam.ai/
SARVAM_API_KEY=your_sarvam_api_key_here

# Groq API Key (required for high-speed LLM fallback)
# Get your free key from: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here
`

### 5. Run the Application
`ash
streamlit run app.py
`

The app will open automatically in your browser at http://localhost:8501.

---

### Live Project : https://documindragsummarizer.streamlit.app/

## 💻 Tech Stack

* **Frontend**: Streamlit
* **LLM Providers**: Sarvam AI API (sarvam-105b), Groq API (openai/gpt-oss-120b, llama-3.3-70b-versatile)
* **Embeddings**: SentenceTransformers (ll-MiniLM-L6-v2)
* **Vector Store**: ChromaDB
* **Document Parsers**: PyPDF, python-docx
* **Language Support**: English, Hindi (हिन्दी), Marathi (मराठी)

---

## 📂 Project Structure

`
DocuMind_RAG/
├── app.py              # Streamlit web dashboard & UI controls
├── rag_service.py      # Core RAG pipeline, multilingual NLP & LLM router
├── requirements.txt    # Project dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Ignored files (keys, cache, venv)
└── README.md           # Project documentation
`

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
