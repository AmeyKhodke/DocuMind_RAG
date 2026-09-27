# RAG Pipeline from scratch for better understanding 
# Developer - Amey Khodke
import io
import os
import re
from pathlib import Path
from collections import Counter
import chromadb
import requests
import pypdf
import docx
from requests.exceptions import RequestException
from sentence_transformers import SentenceTransformer
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

# Try loading env from multiple possible locations
possible_env_paths = [
    Path(__file__).parent / ".env",
    Path(".env"),
    Path("E:/B TECH IT/Celebal Internship/Assignment_7/.env")
]

for env_path in possible_env_paths:
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        break

def get_sarvam_api_key():
    """Retrieves Sarvam AI API subscription key from environment or Streamlit secrets."""
    api_key = os.getenv("SARVAM_API_KEY") or os.getenv("sarvam_api_key")
    if api_key and api_key.strip() and api_key.strip() != "your_sarvam_api_key_here":
        return api_key.strip()
    
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            for key in ["SARVAM_API_KEY", "sarvam_api_key"]:
                if key in st.secrets and st.secrets[key]:
                    val = str(st.secrets[key]).strip()
                    if val and val != "your_sarvam_api_key_here":
                        return val
            sarvam_sec = st.secrets.get("sarvam")
            if isinstance(sarvam_sec, dict):
                for key in ["api_key", "API_KEY", "subscription_key"]:
                    if key in sarvam_sec and sarvam_sec[key]:
                        return str(sarvam_sec[key]).strip()
    except Exception as e:
        print(f"Error reading streamlit secrets for Sarvam: {e}")
    return None

def get_groq_api_key():
    # 1. Try environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.strip() and api_key.strip() != "your_groq_api_key_here":
        return api_key.strip()
        
    # 2. Try streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            # Check direct keys
            for key in ["GROQ_API_KEY", "groq_api_key"]:
                if key in st.secrets:
                    val = st.secrets[key]
                    if val and str(val).strip() != "your_groq_api_key_here":
                        return str(val).strip()
            # Check nested dict keys e.g. [groq] api_key = "..."
            groq_sec = st.secrets.get("groq")
            if isinstance(groq_sec, dict):
                for key in ["api_key", "API_KEY"]:
                    if key in groq_sec and groq_sec[key]:
                        return str(groq_sec[key]).strip()
            elif hasattr(groq_sec, "get"):
                for key in ["api_key", "API_KEY"]:
                    val = groq_sec.get(key)
                    if val:
                        return str(val).strip()
    except Exception as e:
        print(f"Error reading streamlit secrets: {e}")
        
    return None

# --- MULTILINGUAL VOCABULARY & STOPWORD DICTIONARIES ---

ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
    "yours", "yourself", "yourselves"
}

HINDI_STOPWORDS = {
    "का", "के", "की", "को", "ने", "से", "में", "पर", "और", "तथा", "या", "अथवा", "एवं", 
    "है", "हैं", "था", "थी", "थे", "होगा", "होगी", "होंगे", "यह", "वह", "ये", "वे", 
    "इस", "उस", "इन", "उन", "इसका", "उसका", "इनका", "उनका", "इससे", "उससे", "इनसे", 
    "उनसे", "इसमें", "उसमें", "इनमें", "उनमें", "भी", "ही", "तो", "तक", "भर", "मात्र", 
    "नहीं", "मत", "ना", "कि", "यदि", "अगर", "लेकिन", "किन्तु", "परन्तु", "मगर", "बल्कि", 
    "क्योंकि", "इसलिए", "अतः", "अतःएव", "ताकि", "जिससे", "जब", "तब", "कब", "जहाँ", 
    "वहाँ", "कहाँ", "जैसे", "वैसे", "कैसे", "जो", "सो", "कौन", "क्या", "कोई", "कुछ", 
    "बहुत", "सब", "सारा", "तमाम", "अपना", "अपनी", "अपने", "आप", "स्वयं", "खुद", 
    "हुआ", "हुई", "हुए", "किया", "किये", "किए", "गया", "गयी", "गए", "गई", "रहा", 
    "रही", "रहे", "करता", "करती", "करते", "होना", "होने", "करना", "करने", "जाना", 
    "जाने", "आना", "आने", "द्वारा", "प्रति", "बिना", "साथ", "अंदर", "बाहर", "ऊपर", 
    "नीचे", "बीच", "पहले", "बाद", "सामने", "पीछे", "वाले", "वाली", "वाला"
}

MARATHI_STOPWORDS = {
    "आहे", "आहेत", "नाही", "नाहीत", "होता", "होती", "होते", "होतील", "झाले", "झाली", 
    "झाला", "झालेले", "केले", "केली", "केला", "केलेले", "आणि", "व", "किंवा", "अथवा", 
    "पण", "परंतु", "मात्र", "कारण", "म्हणून", "जर", "तर", "जरी", "तरी", "या", "यां", 
    "त्या", "त्यां", "हे", "ही", "तो", "ती", "ते", "जे", "जी", "ज्या", "ज्यां", 
    "असा", "अशी", "असे", "अशा", "असावे", "अशीच", "असेच", "मध्ये", "वर", "खाली", 
    "मागे", "पुढे", "समोर", "जवळ", "कडून", "मुळे", "साठी", "बद्दल", "विषयी", "प्रमाणे", 
    "सह", "सोबत", "स्वतः", "आपण", "आम्ही", "तुम्ही", "मला", "तुला", "त्याला", "तिला", 
    "त्यांना", "आम्हांला", "तुम्हांला", "आमचे", "तुमचे", "त्याचे", "तिचे", "त्यांचे", 
    "आपले", "काही", "सर्व", "सर्वच", "प्रत्येक", "इतर", "अधिक", "फार", "खूप", 
    "जास्त", "कमी", "न", "ना", "च", "सुद्धा", "देखील", "मग", "तेव्हा", "आता", 
    "कधी", "कुठे", "कसे", "का", "काय", "कोण", "कोठे", "कशाला", "येणे", "जाणे", 
    "करणे", "होणे", "देणे", "घेणे", "करून", "देऊन", "घेऊन", "जाऊन", "येऊन", 
    "इत्यादी", "वगैरे"
}

def get_stopwords_for_lang(lang_code):
    if lang_code == "hi":
        return HINDI_STOPWORDS
    elif lang_code == "mr":
        return MARATHI_STOPWORDS
    return ENGLISH_STOPWORDS

# --- LANGUAGE DETECTION ENGINE ---

def detect_language(text):
    """
    Robust zero-dependency detector distinguishing English ('en'), Hindi ('hi'), and Marathi ('mr').
    Returns dict: {"code": "en"|"hi"|"mr", "name": "English"|"Hindi"|"Marathi", "script": "Latin"|"Devanagari"}
    """
    if not text or not text.strip():
        return {"code": "en", "name": "English", "script": "Latin"}
    
    sample = text[:15000]
    latin_count = len(re.findall(r'[a-zA-Z]', sample))
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', sample))
    
    if devanagari_count == 0 or (latin_count > 0 and latin_count / (latin_count + devanagari_count) > 0.65):
        return {"code": "en", "name": "English", "script": "Latin"}
    
    lla_count = len(re.findall(r'\u0933', sample))
    tokens = [t.strip() for t in re.findall(r'[\u0900-\u097F]+', sample.lower()) if len(t.strip()) > 1]
    token_counts = Counter(tokens)
    
    marathi_markers = {
        "आहे", "आहेत", "नाही", "नाहीत", "झाले", "झाली", "झाला", "केले", "केली", "केला", 
        "आणि", "म्हणून", "त्यांच्या", "त्यांना", "त्याचे", "तिचे", "असा", "असे", "अशी", 
        "होता", "होती", "होते", "मध्ये", "करणे", "करून", "घेऊन", "जाऊन", "येऊन", 
        "यांचे", "त्यांचे", "आपल्या", "काही", "विशेष", "तसेच", "वगैरे", "इत्यादी", "शाळा", "वेळ"
    }
    
    hindi_markers = {
        "है", "हैं", "था", "थी", "थे", "होगा", "होगी", "होंगे", "और", "तथा", "लेकिन", 
        "किन्तु", "क्योंकि", "इसलिए", "किया", "किए", "किये", "गया", "गए", "गई", 
        "होता", "होती", "होते", "रहा", "रही", "रहे", "सकता", "सकती", "सकते", 
        "अपना", "अपनी", "अपने", "इसका", "उसका", "इनका", "उनका", "द्वारा", "वाले", "वाली"
    }
    
    marathi_score = (lla_count * 4) + sum(token_counts.get(w, 0) * 2 for w in marathi_markers)
    hindi_score = sum(token_counts.get(w, 0) * 2 for w in hindi_markers)
    
    if marathi_score > hindi_score:
        return {"code": "mr", "name": "Marathi", "script": "Devanagari"}
    else:
        return {"code": "hi", "name": "Hindi", "script": "Devanagari"}

def get_hf_token():
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            for key in ["HF_TOKEN", "hf_token"]:
                if key in st.secrets and st.secrets[key]:
                    return st.secrets[key]
    except Exception:
        pass
    return None

# --- GLOBAL INITIALIZATIONS ---
EMBEDDING_MODEL = None

# Initialize ChromaDB globally so it stays persistent across pipeline calls
CHROMA_CLIENT = chromadb.Client()

def document_loader(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    text = ""
    if file_path.suffix.lower() == ".pdf":
        document = pypdf.PdfReader(file_path)
        for page in document.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    elif file_path.suffix.lower() == ".txt":
        text = file_path.read_text(encoding="utf-8")
    elif file_path.suffix.lower() == ".docx":
        document = docx.Document(str(file_path))
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return text

def chunk_text(text, chunk_size=1000, overlap=150):
    if not text:
        return []
        
    chunks = []
    # Normalize line breaks
    text = re.sub(r'\r\n', '\n', text)
    
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:].strip())
            break
            
        # Look for clean split points (newline, danda, sentence end, or space) in the last 150 characters
        split_candidates = [
            text.rfind('\n', start + chunk_size - 100, end),
            text.rfind('। ', start + chunk_size - 100, end),
            text.rfind('. ', start + chunk_size - 100, end),
            text.rfind(' ', start + chunk_size - 50, end)
        ]
        
        split_point = -1
        for candidate in split_candidates:
            if candidate != -1:
                split_point = candidate
                break
                
        if split_point == -1:
            split_point = end
            
        chunks.append(text[start:split_point].strip())
        
        # Advance the window back by overlap amount
        start = split_point - overlap
        if start < 0:
            start = 0
        # Prevent potential infinite loops if split_point doesn't advance
        if start >= split_point:
            start = split_point + 1
            
    return [c for c in chunks if c.strip()]

def load_text_from_document(document):
    if isinstance(document, (str, Path)):
        file_path = Path(document)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        text = ""
        total_pages = 1
        if file_path.suffix.lower() == ".pdf":
            pdf_reader = pypdf.PdfReader(file_path)
            total_pages = len(pdf_reader.pages)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        elif file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8")
            total_pages = 1
        elif file_path.suffix.lower() == ".docx":
            doc = docx.Document(str(file_path))
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            total_pages = 1
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        return text, total_pages

    if hasattr(document, "read") and hasattr(document, "name"):
        suffix = Path(document.name).suffix.lower()
        content_bytes = document.read()
        if hasattr(document, "seek"):
            document.seek(0)
            
        text = ""
        total_pages = 1
        if suffix == ".pdf":
            pdf_reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            total_pages = len(pdf_reader.pages)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif suffix == ".txt":
            text = content_bytes.decode("utf-8", errors="ignore")
            total_pages = 1
        elif suffix == ".docx":
            doc = docx.Document(io.BytesIO(content_bytes))
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            total_pages = 1
        else:
            raise ValueError("Unsupported document input type")
        return text, total_pages
    raise ValueError("Unsupported document input type")

def get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("Loading embedding model...")
        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return EMBEDDING_MODEL

def create_lightweight_embedding(text, dimensions=384):
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return [0.0] * dimensions
    vector = [0.0] * dimensions
    for token in tokens:
        index = abs(hash(token)) % dimensions
        vector[index] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return [0.0] * dimensions
    return [value / norm for value in vector]

def embedding_generation(chunks):
    token = get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    try:
        embedding_model = get_embedding_model()
        # Enforce list formatting for SentenceTransformer compatibility
        if isinstance(chunks, str):
            chunks = [chunks]
        return embedding_model.encode(chunks, show_progress_bar=False)
    except Exception as exc:
        print(f"Embedding model failed ({exc}). Using lightweight fallback embeddings.")
        if isinstance(chunks, str):
            chunks = [chunks]
        return [create_lightweight_embedding(chunk) for chunk in chunks]

def vector_store_creation(chunks, embeddings):
    collection = CHROMA_CLIENT.get_or_create_collection("document_embeddings")
    
    # Safely flush old entries to prevent mixed context from prior runs
    try:
        existing = collection.get()
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    embedding_values = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(embedding_values))],
        embeddings=embedding_values,
        documents=chunks
    )
    return collection

def query_processing(query, vector_store):
    query_embedding = embedding_generation(query)
    # Ensure nested list structure format required by ChromaDB query engine
    if isinstance(query_embedding, list) and not isinstance(query_embedding[0], list):
        query_embeddings = [query_embedding]
    else:
        query_embeddings = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
        
    try:
        all_ids = vector_store.get()['ids']
        total_chunks = len(all_ids)
    except Exception:
        total_chunks = 3
        
    if total_chunks == 0:
        return {"documents": []}
        
    # If the document is small to medium (<= 25 chunks), retrieve all chunks to give the LLM full context.
    # Otherwise, retrieve the top 10 relevant chunks.
    n_results = total_chunks if total_chunks <= 25 else 10
    n_results = min(n_results, total_chunks)
    if n_results <= 0:
        n_results = 1
        
    return vector_store.query(query_embeddings=query_embeddings, n_results=n_results)

def clean_context_text(text):
    cleaned = re.sub(r"[\r\n]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\bPage\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpp\.\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def split_sentences(text):
    """Splits text into sentences supporting Latin (.!?) and Devanagari (।॥) sentence terminators."""
    if not text:
        return []
    cleaned = clean_context_text(text)
    raw_sentences = re.split(r"(?<=[.!?।॥])\s+", cleaned)
    return [s.strip() for s in raw_sentences if len(s.strip()) >= 15]

def extract_context(results):
    documents = results.get("documents", [])
    if not documents:
        return ""
    flattened = []
    for item in documents:
        if isinstance(item, list):
            flattened.extend(str(sub) for sub in item if sub)
        elif item:
            flattened.append(str(item))
    return clean_context_text(" ".join(flattened))

# --- LANGUAGE-AWARE KEYWORD & EXTRACTIVE RANKING ---

def extract_language_aware_keywords(text, lang="en", top_n=15):
    """Extracts informative topical keywords by filtering language-specific stopwords."""
    if not text:
        return []
    stopwords = get_stopwords_for_lang(lang)
    
    if lang in ["hi", "mr"]:
        tokens = re.findall(r'[\u0900-\u097F]+', text.lower())
        meaningful_tokens = [t for t in tokens if len(t) >= 2 and t not in stopwords]
    else:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        meaningful_tokens = [t for t in tokens if t not in stopwords]
        
    counts = Counter(meaningful_tokens)
    return [word for word, _ in counts.most_common(top_n)]

def rank_sentences_for_compression(text, lang="en", max_chars=4500):
    """
    Selects the most informative sentences using language-specific stopwords,
    keyword frequencies, and position weighting. Preserves document chronological flow.
    """
    sentences = split_sentences(text)
    if not sentences:
        return text[:max_chars]
        
    total_len = sum(len(s) for s in sentences)
    if total_len <= max_chars:
        return " ".join(sentences)
        
    keywords = extract_language_aware_keywords(text, lang, top_n=25)
    keyword_set = set(keywords)
    total_sentences = len(sentences)
    
    scored_sentences = []
    for idx, sentence in enumerate(sentences):
        s_lower = sentence.lower()
        if lang in ["hi", "mr"]:
            words = set(re.findall(r'[\u0900-\u097F]+', s_lower))
        else:
            words = set(re.findall(r'\b[a-zA-Z]+\b', s_lower))
            
        overlap = len(words.intersection(keyword_set))
        
        position_weight = 1.0
        if idx < max(3, int(total_sentences * 0.12)):
            position_weight = 1.6
        elif idx >= total_sentences - max(2, int(total_sentences * 0.10)):
            position_weight = 1.3
            
        score = (overlap + 1) * position_weight
        scored_sentences.append((score, idx, sentence))
        
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    selected = []
    current_length = 0
    for score, idx, sent in scored_sentences:
        if current_length + len(sent) > max_chars:
            continue
        selected.append((idx, sent))
        current_length += len(sent) + 1
        if current_length >= max_chars * 0.95:
            break
            
    selected.sort(key=lambda x: x[0])
    return " ".join(s for _, s in selected)

def build_fallback_summary(text, lang="en", target_lang=None):
    """Generates a structured extractive summary respecting language conventions."""
    if not text:
        return "No text available to summarize."
        
    sentences = split_sentences(text)
    if not sentences:
        return text[:400] + "..."
        
    intro = sentences[:3]
    conclusion = sentences[-2:] if len(sentences) > 5 else []
    
    keywords = extract_language_aware_keywords(text, lang, top_n=6)
    keyword_set = set(keywords)
    
    body_sentences = []
    if len(sentences) > 5:
        middle_sentences = sentences[3:-2]
        scored = []
        for s in middle_sentences:
            s_lower = s.lower()
            if lang in ["hi", "mr"]:
                words = set(re.findall(r'[\u0900-\u097F]+', s_lower))
            else:
                words = set(re.findall(r'\b[a-zA-Z]+\b', s_lower))
            hits = len(words.intersection(keyword_set))
            scored.append((hits, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        body_sentences = [s for hits, s in scored[:3] if hits > 0]
        
    effective_lang = target_lang or lang
    if effective_lang == "hi":
        key_label = "\n\n**मुख्य बिंदु:**"
        concl_label = "\n\n**निष्कर्ष:**"
    elif effective_lang == "mr":
        key_label = "\n\n**महत्त्वाचे मुद्दे:**"
        concl_label = "\n\n**निष्कर्ष:**"
    else:
        key_label = "\n\n**Key Details:**"
        concl_label = "\n\n**Conclusion:**"
        
    parts = [" ".join(intro)]
    if body_sentences:
        parts.append(key_label)
        parts.append("\n- " + "\n- ".join(body_sentences))
    if conclusion:
        parts.append(concl_label)
        parts.append(" ".join(conclusion))
        
    return " ".join(parts)

def build_fallback_answer(query, context, lang="en"):
    if not context:
        return "I don't have enough context to answer that question."
    sentences = split_sentences(context)
    if not sentences:
        return clean_context_text(context)[:300]
        
    if lang in ["hi", "mr"]:
        query_terms = set(re.findall(r'[\u0900-\u097F]+', query.lower()))
    else:
        query_terms = set(re.findall(r'\w+', query.lower()))
        
    stopwords = get_stopwords_for_lang(lang)
    query_terms = {t for t in query_terms if t not in stopwords and len(t) >= 2}
    
    scored_sentences = []
    for sentence in sentences:
        s_lower = sentence.lower()
        if lang in ["hi", "mr"]:
            words = set(re.findall(r'[\u0900-\u097F]+', s_lower))
        else:
            words = set(re.findall(r'\w+', s_lower))
        score = len(words.intersection(query_terms))
        scored_sentences.append((score, sentence))
        
    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = [s for score, s in scored_sentences if score > 0][:3]
    return " ".join(selected) if selected else sentences[0]

# --- LLM API CLIENTS (SARVAM AI & GROQ) ---

def call_sarvam_api(prompt, temperature=0.0, timeout=45):
    """
    Invokes Sarvam AI Chat Completions API (optimized for Indic languages: Hindi, Marathi, English).
    """
    api_key = get_sarvam_api_key()
    if not api_key:
        raise ValueError("SARVAM_API_KEY not configured.")
        
    url = "https://api.sarvam.ai/v1/chat/completions"
    headers = {
        "api-subscription-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    models = ["sarvam-105b"]
    last_err = None
    for model in models:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 2048
            }
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            res_json = response.json()
            choice = res_json["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content")
            if content and content.strip():
                return content.strip()
            # If reasoning model hit length limit before writing final content, use reasoning content
            reasoning = msg.get("reasoning_content")
            if reasoning and reasoning.strip():
                return reasoning.strip()
            raise ValueError(f"Empty content returned by Sarvam API (finish_reason: {choice.get('finish_reason')})")
        except Exception as e:
            last_err = e
            print(f"Sarvam AI API error for {model}: {e}")
            continue
    raise last_err

def call_groq_api(prompt, temperature=0.0):
    """Invokes Groq API with robust Indic/Multilingual model fallbacks."""
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment or streamlit secrets.")
        
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
    ]
    last_err = None
    for model in models:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 1024
            }
            response = requests.post(url, headers=headers, json=payload, timeout=25)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            print(f"Groq API error for model {model}: {e}")
            continue
    raise last_err

def generate_llm_response(prompt, temperature=0.0, timeout=35, prefer_sarvam=False):
    """
    Multi-tier LLM generation router:
    1. Sarvam AI API (if prefer_sarvam or Indic language involved and key present)
    2. Groq API (High performance multilingual fallback)
    3. Sarvam AI API (if not tried yet)
    4. Local Ollama (gemma:2b)
    """
    sarvam_key = get_sarvam_api_key()
    groq_key = get_groq_api_key()
    
    # Tier 1: Prefer Sarvam for Indic queries if configured
    if prefer_sarvam and sarvam_key:
        try:
            return call_sarvam_api(prompt, temperature=temperature, timeout=timeout), "Sarvam AI API"
        except Exception as e:
            print(f"Sarvam AI failed ({e}). Falling back to Groq API.")
            
    # Tier 2: Groq API
    if groq_key:
        try:
            return call_groq_api(prompt, temperature=temperature), "LLM Groq API"
        except Exception as e:
            print(f"Groq API failed ({e}).")
            
    # Tier 2b: Try Sarvam if it wasn't tried yet
    if not prefer_sarvam and sarvam_key:
        try:
            return call_sarvam_api(prompt, temperature=temperature, timeout=timeout), "Sarvam AI API"
        except Exception as e:
            print(f"Sarvam AI backup try failed ({e}).")
            
    # Tier 3: Local Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma:2b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()["response"].strip(), "Ollama (Local Fallback)"
    except Exception as e:
        raise RuntimeError(f"All LLM backends (Sarvam, Groq, Ollama) failed: {e}")

# --- RESOLVING TARGET LANGUAGE ---

def resolve_target_language_name(target_language, source_lang_code):
    """Maps user selection or code to human-readable language name and code."""
    target_str = str(target_language or "source").lower().strip()
    if target_str in ["source", "same as document", "same"]:
        code = source_lang_code
    elif "marathi" in target_str or target_str == "mr":
        code = "mr"
    elif "hindi" in target_str or target_str == "hi":
        code = "hi"
    elif "english" in target_str or target_str == "en":
        code = "en"
    else:
        code = source_lang_code
        
    names = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    return code, names.get(code, "English")

# --- CONTEXT RETRIEVAL & QUERY ANSWERING ---

def context_retrieval(query, context, ingested_data=None, target_language="source"):
    source_info = (ingested_data or {}).get("language", {"code": "en", "name": "English"})
    source_code = source_info.get("code", "en")
    source_name = source_info.get("name", "English")
    
    target_code, target_name = resolve_target_language_name(target_language, source_code)
    is_indic = target_code in ["hi", "mr"] or source_code in ["hi", "mr"]
    
    metadata_str = ""
    if ingested_data:
        file_name = ingested_data.get("file_name", "Uploaded Document")
        total_pages = ingested_data.get("total_pages", "N/A")
        total_chunks = len(ingested_data.get("chunks", []))
        metadata_str = f"Document Metadata:\n- File Name: {file_name}\n- Total Pages: {total_pages}\n- Total Chunks: {total_chunks}\n- Document Language: {source_name}\n\n"

    prompt = f"""Instructions: You are a helpful, factual multilingual document assistant.
Document Source Language: {source_name}
Required Response Language: {target_name}

Answer the following Question based strictly on the provided Context and Document Metadata.
Provide your response entirely in {target_name}.
If the provided information does not contain the answer, reply in {target_name} stating that the information is not available in the document.

{metadata_str}Context: {context}

Question: {query}
Answer in {target_name}:"""

    try:
        return generate_llm_response(prompt, temperature=0.0, timeout=30, prefer_sarvam=is_indic)
    except Exception as e:
        print(f"LLM generation failed ({e}). Using language-aware fallback scoring.")
        return build_fallback_answer(query, context, lang=source_code), "Local Fallback (Heuristic Scoring)"

# --- SUMMARIZATION ---

def summarize_document(ingested_data, target_language="source"):
    """
    Summarizes document using language-aware sentence compression,
    with cross-lingual generation in the user's selected target language (English, Hindi, Marathi).
    """
    if not ingested_data or not ingested_data.get("text"):
        return "No document text available to summarize.", "System Info"
        
    text = ingested_data["text"]
    source_info = ingested_data.get("language", detect_language(text))
    source_code = source_info.get("code", "en")
    source_name = source_info.get("name", "English")
    
    target_code, target_name = resolve_target_language_name(target_language, source_code)
    is_indic = target_code in ["hi", "mr"] or source_code in ["hi", "mr"]
    
    # Apply language-aware extractive compression for large documents
    compressed_text = rank_sentences_for_compression(text, lang=source_code, max_chars=4500)
    
    prompt = f"""Instructions: You are an expert multilingual document assistant specializing in English, Hindi, and Marathi.
Source Document Language: {source_name}
Target Output Language: {target_name}

Provide a comprehensive, well-structured summary of the following document.
CRITICAL RULE: The summary MUST BE WRITTEN ENTIRELY IN {target_name.upper()}.
Accurately capture the key points, core facts, and overall conclusions.

Structure the summary cleanly with:
- Overview
- Key Points (bulleted)
- Conclusion

Document Content:
{compressed_text}

Summary in {target_name}:"""

    try:
        return generate_llm_response(prompt, temperature=0.2, timeout=45, prefer_sarvam=is_indic)
    except Exception as e:
        print(f"LLM summarization failed ({e}). Using language-aware fallback extractive summary.")
        fallback = build_fallback_summary(compressed_text, lang=source_code, target_lang=target_code)
        return fallback, "Fallback (Extractive Heuristics)"

# --- INGESTION & PIPELINE ---

def ingest_document(document):
    """
    Parses document, detects language, chunks it, generates embeddings, stores them in ChromaDB,
    and returns a metadata dictionary with language details.
    """
    text, total_pages = load_text_from_document(document)
    if not text or not text.strip():
        raise ValueError("Unable to extract text from the uploaded document.")
        
    lang_info = detect_language(text)
    chunks = chunk_text(text, chunk_size=1000, overlap=150)
    if not chunks:
        raise ValueError("No text chunks generated from the document.")
        
    embeddings = embedding_generation(chunks)
    vector_store = vector_store_creation(chunks, embeddings)
    
    if hasattr(document, "name"):
        file_name = document.name
    elif isinstance(document, (str, Path)):
        file_name = Path(document).name
    else:
        file_name = "Uploaded Document"
        
    return {
        "file_name": file_name,
        "total_pages": total_pages,
        "language": lang_info,
        "chunks": chunks,
        "collection_name": "document_embeddings",
        "text": text
    }

def answer_query(ingested_data, query, target_language="source"):
    """
    Queries the vector store and gets context to retrieve the answer in requested target language.
    """
    if not ingested_data:
        return "Please upload a document first.", "System Info"
        
    collection_name = ingested_data.get("collection_name", "document_embeddings")
    collection = CHROMA_CLIENT.get_or_create_collection(collection_name)
    results = query_processing(query, collection)
    context = extract_context(results)
    
    return context_retrieval(query, context, ingested_data, target_language=target_language)

def pipeline(document, query=None, target_language="source"):
    ingested = ingest_document(document)
    if query is None:
        return ingested
    return answer_query(ingested, query, target_language=target_language)

