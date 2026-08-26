# RAG Pipeline from scratch for better understanding 
# Developer - Amey Khodke (Optimized & Multilingual Enhanced)
import io
import os
import re
import json
import hashlib
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import chromadb
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pypdf
import docx
from sentence_transformers import SentenceTransformer
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

# Safe Streamlit decorator wrapper
try:
    import streamlit as st
    cache_resource = st.cache_resource
except Exception:
    def cache_resource(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

# Try loading env from multiple possible locations
possible_env_paths = [
    Path(__file__).parent / ".env",
    Path(".env"),
    Path("E:/B TECH IT/Celebal Internship/Assignment_7/.env")
]

for env_path in possible_env_paths:
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        break

# Module-level precompiled regex patterns
RE_LINE_BREAKS = re.compile(r'\r\n')
RE_WORD_TOKENS = re.compile(r'\w+', re.IGNORECASE)
RE_MULTIPLE_NEWLINES = re.compile(r'[\r\n]+')
RE_MULTIPLE_SPACES = re.compile(r'\s+')
RE_PAGE_NUMBERS = re.compile(r'\bPage\s*\d+\b', re.IGNORECASE)
RE_PP_NUMBERS = re.compile(r'\bpp\.\s*\d+\b', re.IGNORECASE)
RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
RE_LONG_WORDS = re.compile(r'\b\w{5,}\b', re.IGNORECASE)

# Reuse a single requests.Session() with connection pooling and retries
def _create_http_session():
    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

HTTP_SESSION = _create_http_session()

# In-memory query cache & disk cache dir
QUERY_CACHE = {}
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Language Mapping Dict (Extensible for 4th/5th languages)
LANGUAGE_CODES = {
    "English": "en-IN",
    "हिंदी (Hindi)": "hi-IN",
    "मराठी (Marathi)": "mr-IN"
}

def get_groq_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            for key in ["GROQ_API_KEY", "groq_api_key"]:
                if key in st.secrets and st.secrets[key]:
                    return st.secrets[key]
            groq_sec = st.secrets.get("groq")
            if isinstance(groq_sec, dict):
                for key in ["api_key", "API_KEY"]:
                    if key in groq_sec and groq_sec[key]:
                        return groq_sec[key]
            elif hasattr(groq_sec, "get"):
                for key in ["api_key", "API_KEY"]:
                    val = groq_sec.get(key)
                    if val:
                        return val
    except Exception as e:
        print(f"Error reading streamlit secrets: {e}")
    return None

def get_sarvam_api_key():
    api_key = os.getenv("SARVAM_API_KEY")
    if api_key:
        return api_key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            for key in ["SARVAM_API_KEY", "sarvam_api_key"]:
                if key in st.secrets and st.secrets[key]:
                    return st.secrets[key]
            sarvam_sec = st.secrets.get("sarvam")
            if isinstance(sarvam_sec, dict):
                for key in ["api_key", "API_KEY"]:
                    if key in sarvam_sec and sarvam_sec[key]:
                        return sarvam_sec[key]
            elif hasattr(sarvam_sec, "get"):
                for key in ["api_key", "API_KEY"]:
                    val = sarvam_sec.get(key)
                    if val:
                        return val
    except Exception as e:
        print(f"Error reading streamlit secrets: {e}")
    return None

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

# Singleton / Cached Model & Vector Store Client
@cache_resource
def get_embedding_model():
    print("Loading embedding model (singleton)...")
    return SentenceTransformer("all-MiniLM-L6-v2")

@cache_resource
def get_chroma_client():
    print("Initializing ChromaDB Client (singleton)...")
    return chromadb.Client()

def clear_query_cache():
    global QUERY_CACHE
    QUERY_CACHE.clear()

def _extract_pdf_page(args):
    content_bytes, page_idx = args
    try:
        reader = pypdf.PdfReader(io.BytesIO(content_bytes))
        if page_idx < len(reader.pages):
            return reader.pages[page_idx].extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF page {page_idx}: {e}")
    return ""

def extract_pdf_text_parallel(content_bytes):
    reader = pypdf.PdfReader(io.BytesIO(content_bytes))
    total_pages = len(reader.pages)
    if total_pages <= 1:
        text = reader.pages[0].extract_text() if total_pages == 1 else ""
        return text or "", total_pages
    
    tasks = [(content_bytes, i) for i in range(total_pages)]
    max_workers = min(8, total_pages)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pages_text = list(executor.map(_extract_pdf_page, tasks))
    return "\n".join(t for t in pages_text if t.strip()), total_pages

def document_loader(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    text = ""
    if file_path.suffix.lower() == ".pdf":
        content_bytes = file_path.read_bytes()
        text, _ = extract_pdf_text_parallel(content_bytes)
    elif file_path.suffix.lower() == ".txt":
        text = file_path.read_text(encoding="utf-8")
    elif file_path.suffix.lower() == ".docx":
        document = docx.Document(str(file_path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return text

def chunk_text(text, chunk_size=1000, overlap=150):
    if not text:
        return []
        
    chunks = []
    text = RE_LINE_BREAKS.sub('\n', text)
    
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:].strip())
            break
            
        split_candidates = [
            text.rfind('\n', start + chunk_size - 100, end),
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
        
        start = split_point - overlap
        if start < 0:
            start = 0
        if start >= split_point:
            start = split_point + 1
            
    return [c for c in chunks if c.strip()]

def load_text_from_document(document):
    if isinstance(document, (str, Path)):
        file_path = Path(document)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == ".pdf":
            content_bytes = file_path.read_bytes()
            return extract_pdf_text_parallel(content_bytes)
        elif file_path.suffix.lower() == ".txt":
            return file_path.read_text(encoding="utf-8"), 1
        elif file_path.suffix.lower() == ".docx":
            doc = docx.Document(str(file_path))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs), 1
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    if hasattr(document, "read") and hasattr(document, "name"):
        suffix = Path(document.name).suffix.lower()
        content_bytes = document.read()
        if hasattr(document, "seek"):
            document.seek(0)
            
        if suffix == ".pdf":
            return extract_pdf_text_parallel(content_bytes)
        elif suffix == ".txt":
            return content_bytes.decode("utf-8", errors="ignore"), 1
        elif suffix == ".docx":
            doc = docx.Document(io.BytesIO(content_bytes))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs), 1
        else:
            raise ValueError("Unsupported document input type")
    raise ValueError("Unsupported document input type")

def create_lightweight_embedding(text, dimensions=384):
    tokens = RE_WORD_TOKENS.findall(text.lower())
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
        if isinstance(chunks, str):
            chunks = [chunks]
        return embedding_model.encode(chunks, batch_size=32, show_progress_bar=False)
    except Exception as exc:
        print(f"Embedding model failed ({exc}). Using lightweight fallback embeddings.")
        if isinstance(chunks, str):
            chunks = [chunks]
        return [create_lightweight_embedding(chunk) for chunk in chunks]

def vector_store_creation(chunks, embeddings):
    client = get_chroma_client()
    collection_name = "document_embeddings"
    
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(collection_name)
    embedding_values = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(embedding_values))],
        embeddings=embedding_values,
        documents=chunks
    )
    return collection

def query_processing(query, vector_store):
    query_embedding = embedding_generation(query)
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
        
    n_results = total_chunks if total_chunks <= 25 else 10
    n_results = min(n_results, total_chunks)
    if n_results <= 0:
        n_results = 1
        
    return vector_store.query(query_embeddings=query_embeddings, n_results=n_results)

def clean_context_text(text):
    cleaned = RE_MULTIPLE_NEWLINES.sub(" ", text)
    cleaned = RE_MULTIPLE_SPACES.sub(" ", cleaned).strip()
    cleaned = RE_PAGE_NUMBERS.sub("", cleaned)
    cleaned = RE_PP_NUMBERS.sub("", cleaned)
    return cleaned.strip()

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

def call_groq_api(prompt, temperature=0.0, stream=False):
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment or streamlit secrets.")
        
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    models_env = os.getenv("GROQ_MODELS")
    if models_env:
        models = [m.strip() for m in models_env.split(",") if m.strip()]
    else:
        models = [
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
            "qwen/qwen3.6-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ]
        
    timeout = int(os.getenv("GROQ_TIMEOUT", "12"))
    last_err = None
    
    for model in models:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 1024,
                "stream": stream
            }
            if stream:
                response = HTTP_SESSION.post(url, headers=headers, json=payload, timeout=timeout, stream=True)
                response.raise_for_status()
                
                def generate_chunks():
                    for line in response.iter_lines():
                        if not line:
                            continue
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0]["delta"]
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except Exception:
                                pass
                return generate_chunks()
            else:
                response = HTTP_SESSION.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                res_json = response.json()
                return res_json["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            print(f"Groq API error for model {model}: {e}")
            continue
    raise last_err

def generate_llm_response(prompt, temperature=0.0, timeout=30, stream=False):
    # Tier 1: Try Groq API
    api_key = get_groq_api_key()
    if api_key:
        try:
            return call_groq_api(prompt, temperature=temperature, stream=stream), "LLM Groq API"
        except Exception as e:
            print(f"Groq API failed: {e}. Falling back to local Ollama.")
    else:
        print("GROQ_API_KEY not configured. Trying local Ollama.")
        
    # Tier 2: Try local Ollama
    try:
        if stream:
            response = HTTP_SESSION.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma:2b",
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": temperature}
                },
                timeout=timeout,
                stream=True
            )
            response.raise_for_status()
            def generate_ollama_chunks():
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode('utf-8'))
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                    except Exception:
                        pass
            return generate_ollama_chunks(), "Ollama (Local Fallback)"
        else:
            response = HTTP_SESSION.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma:2b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()["response"].strip(), "Ollama (Local Fallback)"
    except Exception as e:
        raise RuntimeError(f"Ollama local service failed or not reachable: {e}")

def translate_with_sarvam(text, target_lang_code):
    api_key = get_sarvam_api_key()
    if not api_key:
        raise ValueError("SARVAM_API_KEY not found.")
        
    url = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }
    
    max_chunk_len = 1000
    if len(text) <= max_chunk_len:
        text_chunks = [text]
    else:
        paragraphs = text.split("\n")
        text_chunks = []
        curr = ""
        for p in paragraphs:
            if len(curr) + len(p) + 1 <= max_chunk_len:
                curr = (curr + "\n" + p).strip() if curr else p
            else:
                if curr:
                    text_chunks.append(curr)
                if len(p) <= max_chunk_len:
                    curr = p
                else:
                    sents = RE_SENTENCE_SPLIT.split(p)
                    for s in sents:
                        if len(curr) + len(s) + 1 <= max_chunk_len:
                            curr = (curr + " " + s).strip() if curr else s
                        else:
                            if curr:
                                text_chunks.append(curr)
                            curr = s[:max_chunk_len]
        if curr:
            text_chunks.append(curr)
            
    translated_chunks = []
    for chunk in text_chunks:
        if not chunk.strip():
            translated_chunks.append(chunk)
            continue
        payload = {
            "input": chunk,
            "source_language_code": "en-IN",
            "target_language_code": target_lang_code,
            "model": "sarvam-translate:v1"
        }
        resp = HTTP_SESSION.post(url, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        res_json = resp.json()
        translated_text = res_json.get("translated_text", chunk)
        translated_chunks.append(translated_text)
        
    return "\n".join(translated_chunks)

def translate_with_llm(text, target_lang_name):
    prompt = f"""Translate the following text into {target_lang_name}. Preserve meaning, tone, and any factual details exactly. Return only the translated text, no preamble.

Text:
{text}"""
    translated_text, source = generate_llm_response(prompt, temperature=0.1, timeout=30, stream=False)
    return translated_text, source

def translate_text(text, target_lang="English"):
    if not text or target_lang in ["English", "en"]:
        return text, None
        
    lang_code = LANGUAGE_CODES.get(target_lang)
    if not lang_code:
        return text, None
        
    # Tier 1: Sarvam AI Translate API
    sarvam_key = get_sarvam_api_key()
    if sarvam_key:
        try:
            translated = translate_with_sarvam(text, lang_code)
            return translated, f"Sarvam AI ({target_lang})"
        except Exception as e:
            print(f"Sarvam translation failed: {e}. Falling back to LLM translation.")
    else:
        print("SARVAM_API_KEY not configured. Falling back to LLM translation.")
        
    # Tier 2: LLM Fallback (Groq / Ollama)
    try:
        translated, source = translate_with_llm(text, target_lang)
        return translated, f"{source} ({target_lang})"
    except Exception as e:
        print(f"LLM translation failed: {e}. Returning original text with note.")
        
    # Tier 3: Return original text untranslated with note
    return text, f"Translation Unavailable ({target_lang})"

def context_retrieval(query, context, ingested_data=None, stream=False):
    metadata_str = ""
    if ingested_data:
        file_name = ingested_data.get("file_name", "Uploaded Document")
        total_pages = ingested_data.get("total_pages", "N/A")
        total_chunks = len(ingested_data.get("chunks", []))
        metadata_str = f"Document Metadata:\n- File Name: {file_name}\n- Total Pages: {total_pages}\n- Total Chunks: {total_chunks}\n\n"

    prompt = f"""Instructions: You are a helpful and factual document assistant. Answer the Question based on the provided Context and Document Metadata.
Be direct and detailed in your answer. If the provided information does not contain the answer, reply with: "I don't know based on the given information."

{metadata_str}Context: {context}

Question: {query}
Answer:"""

    try:
        return generate_llm_response(prompt, temperature=0.0, timeout=30, stream=stream)
    except Exception as e:
        print(f"LLM generation failed ({e}). Attempting internal fallback scoring algorithm.")
        fallback_ans = build_fallback_answer(query, context)
        if stream:
            def single_chunk():
                yield fallback_ans
            return single_chunk(), "Local Fallback (Heuristic Scoring)"
        return fallback_ans, "Local Fallback (Heuristic Scoring)"

def build_fallback_answer(query, context):
    if not context:
        return "I don't have enough context to answer that question."
    cleaned_context = clean_context_text(context)
    sentences = RE_SENTENCE_SPLIT.split(cleaned_context)
    query_terms = RE_WORD_TOKENS.findall(query.lower())
    
    scored_sentences = []
    for sentence in sentences:
        if len(sentence) < 25:
            continue
        sentence_lower = sentence.lower()
        score = sum(1 for term in query_terms if term in sentence_lower)
        scored_sentences.append((score, sentence.strip()))
        
    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = [s for score, s in scored_sentences if score > 0][:3]
    return " ".join(selected) if selected else cleaned_context[:300]

# --- EXECUTION PIPELINE ---
def ingest_document(document):
    clear_query_cache()
    
    if hasattr(document, "read"):
        content_bytes = document.read()
        if hasattr(document, "seek"):
            document.seek(0)
    elif isinstance(document, (str, Path)):
        content_bytes = Path(document).read_bytes()
    else:
        content_bytes = b""
        
    doc_hash = hashlib.sha256(content_bytes).hexdigest() if content_bytes else "unknown"
    cache_path = CACHE_DIR / f"{doc_hash}.json"
    
    file_name = getattr(document, "name", Path(document).name if isinstance(document, (str, Path)) else "Uploaded Document")
    
    if cache_path.exists():
        print(f"Loading document chunks and embeddings from disk cache: {cache_path}")
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            chunks = cached_data["chunks"]
            embeddings = cached_data["embeddings"]
            total_pages = cached_data["total_pages"]
            text = cached_data["text"]
            
            vector_store = vector_store_creation(chunks, embeddings)
            return {
                "doc_hash": doc_hash,
                "file_name": file_name,
                "total_pages": total_pages,
                "chunks": chunks,
                "collection_name": "document_embeddings",
                "text": text
            }
        except Exception as e:
            print(f"Cache load failed: {e}. Rebuilding...")

    text, total_pages = load_text_from_document(document)
    if not text or not text.strip():
        raise ValueError("Unable to extract text from the uploaded document.")
        
    chunks = chunk_text(text, chunk_size=1000, overlap=150)
    if not chunks:
        raise ValueError("No text chunks generated from the document.")
        
    embeddings = embedding_generation(chunks)
    vector_store = vector_store_creation(chunks, embeddings)
    
    try:
        embedding_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        cache_data = {
            "doc_hash": doc_hash,
            "file_name": file_name,
            "total_pages": total_pages,
            "chunks": chunks,
            "embeddings": embedding_list,
            "text": text
        }
        cache_path.write_text(json.dumps(cache_data), encoding="utf-8")
    except Exception as e:
        print(f"Failed to write disk cache: {e}")

    return {
        "doc_hash": doc_hash,
        "file_name": file_name,
        "total_pages": total_pages,
        "chunks": chunks,
        "collection_name": "document_embeddings",
        "text": text
    }

def answer_query(ingested_data, query, target_lang="English", stream=False):
    if not ingested_data:
        return "Please upload a document first.", "System Info"
        
    doc_hash = ingested_data.get("doc_hash", "default")
    cache_key = (doc_hash, query.strip().lower(), target_lang)
    
    if cache_key in QUERY_CACHE:
        return QUERY_CACHE[cache_key]

    client = get_chroma_client()
    collection_name = ingested_data.get("collection_name", "document_embeddings")
    collection = client.get_or_create_collection(collection_name)
    results = query_processing(query, collection)
    context = extract_context(results)
    
    effective_stream = stream if target_lang == "English" else False
    
    res, source = context_retrieval(query, context, ingested_data, stream=effective_stream)
    
    if target_lang != "English":
        if hasattr(res, "__iter__") and not isinstance(res, str):
            res = "".join(list(res))
        translated_res, trans_source = translate_text(res, target_lang)
        if trans_source:
            source = f"{source} → {trans_source}"
        response_tuple = (translated_res, source)
        QUERY_CACHE[cache_key] = response_tuple
        return response_tuple
    else:
        if effective_stream:
            return res, source
        else:
            response_tuple = (res, source)
            QUERY_CACHE[cache_key] = response_tuple
            return response_tuple

def build_fallback_summary(text):
    if not text:
        return "No text to summarize."
    cleaned_text = clean_context_text(text)
    sentences = RE_SENTENCE_SPLIT.split(cleaned_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    if not sentences:
        return text[:300] + "..."
        
    intro = sentences[:3]
    conclusion = sentences[-2:] if len(sentences) > 5 else []
    
    all_words = RE_LONG_WORDS.findall(cleaned_text.lower())
    from collections import Counter
    word_counts = Counter(all_words)
    top_words = [word for word, count in word_counts.most_common(5)]
    
    body_sentences = []
    if len(sentences) > 5:
        middle_sentences = sentences[3:-2]
        scored = []
        for s in middle_sentences:
            score = sum(1 for word in top_words if word in s.lower())
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        body_sentences = [s for score, s in scored[:3] if score > 0]
        
    summary_parts = intro
    if body_sentences:
        summary_parts.append("\n\n**Key Details:**")
        summary_parts.extend(body_sentences)
    if conclusion:
        summary_parts.append("\n\n**Conclusion:**")
        summary_parts.extend(conclusion)
        
    return " ".join(summary_parts)

def summarize_document(ingested_data, target_lang="English"):
    if not ingested_data or not ingested_data.get("text"):
        return "No document text available to summarize.", "System Info"
        
    doc_hash = ingested_data.get("doc_hash", "default")
    cache_key = (doc_hash, "__SUMMARY__", target_lang)
    if cache_key in QUERY_CACHE:
        return QUERY_CACHE[cache_key]

    text = ingested_data["text"]
    
    max_summary_input_chars = 6000
    if len(text) > max_summary_input_chars:
        input_text = text[:4000] + "\n... [text truncated for summarization] ...\n" + text[-2000:]
    else:
        input_text = text

    prompt = f"""Instructions: Provide a concise, comprehensive summary of the following document. Highlight the main topics, key points, and overall conclusion.
Document:
{input_text}

Summary:"""

    try:
        raw_summary, source = generate_llm_response(prompt, temperature=0.3, timeout=45, stream=False)
    except Exception as e:
        print(f"LLM summarization failed ({e}). Using fallback extractive summary.")
        raw_summary, source = build_fallback_summary(text), "Fallback (Extractive Heuristics)"

    if target_lang != "English":
        translated_summary, trans_source = translate_text(raw_summary, target_lang)
        if trans_source:
            source = f"{source} → {trans_source}"
        res_tuple = (translated_summary, source)
    else:
        res_tuple = (raw_summary, source)
        
    QUERY_CACHE[cache_key] = res_tuple
    return res_tuple

def pipeline(document, query=None):
    ingested = ingest_document(document)
    if query is None:
        return ingested
    return answer_query(ingested, query)
