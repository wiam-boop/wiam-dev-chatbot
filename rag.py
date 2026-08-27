"""
محرك RAG (Retrieval-Augmented Generation)
==========================================
يستخرج النص من الملفات (PDF, Word, صور)، يقسّمه لأجزاء صغيرة (chunks)،
يحوّلها لمتجهات (embeddings) عبر نموذج محلي مجاني، ويخزّنها في FAISS.
عند كل سؤال، يبحث عن أقرب الأجزاء معنى للسؤال ويعيدها كسياق للنموذج.
"""

import os
import json
import uuid
import threading

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from pypdf import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract

# ✅ التعديل الأساسي: استخدام /tmp على Railway أو storage محلي عند التطوير
IS_RAILWAY = os.environ.get("RAILWAY_ENVIRONMENT") is not None

if IS_RAILWAY:
    STORAGE_DIR = "/tmp/storage"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    STORAGE_DIR = os.path.join(BASE_DIR, "storage")

UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
INDEX_PATH  = os.path.join(STORAGE_DIR, "kb.index")
META_PATH   = os.path.join(STORAGE_DIR, "kb_meta.json")

os.makedirs(UPLOADS_DIR, exist_ok=True)

# نموذج تضمين متعدد اللغات — يعمل محلياً بدون API
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

CHUNK_SIZE    = 700
CHUNK_OVERLAP = 120
TOP_K         = 4

_lock  = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# استخراج النص من أنواع الملفات المختلفة
# ---------------------------------------------------------------------------

def extract_text(file_path: str, ext: str) -> str:
    ext = ext.lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext in (".docx",):
        return _extract_docx(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        return _extract_image(file_path)
    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"صيغة غير مدعومة: {ext}")


def _extract_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n".join(parts)


def _extract_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text for cell in row.cells)
            if row_text.strip():
                parts.append(row_text)
    return "\n".join(parts)


def _extract_image(file_path: str) -> str:
    image = Image.open(file_path)
    try:
        text = pytesseract.image_to_string(image, lang="ara+eng")
    except Exception:
        text = pytesseract.image_to_string(image)
    return text


# ---------------------------------------------------------------------------
# تقسيم النص لأجزاء (chunking)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# قاعدة المعرفة (Knowledge Base)
# ---------------------------------------------------------------------------

class KnowledgeBase:
    def __init__(self):
        self.index = None
        self.meta  = []
        self._load()

    def _load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            self.meta  = []

    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def add_document(self, text: str, source_name: str, doc_id: str):
        chunks = chunk_text(text)
        if not chunks:
            return 0

        model      = _get_model()
        embeddings = model.encode(chunks, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        with _lock:
            self.index.add(embeddings)
            for i, chunk in enumerate(chunks):
                self.meta.append({
                    "doc_id":   doc_id,
                    "source":   source_name,
                    "chunk_no": i,
                    "text":     chunk,
                })
            self._save()

        return len(chunks)

    def search(self, query: str, top_k=TOP_K):
        if self.index.ntotal == 0:
            return []

        model  = _get_model()
        q_emb  = model.encode([query], normalize_embeddings=True)
        q_emb  = np.array(q_emb, dtype="float32")

        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item          = dict(self.meta[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def list_documents(self):
        seen = {}
        for item in self.meta:
            doc_id = item["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "source": item["source"],
                    "chunks": 0,
                }
            seen[doc_id]["chunks"] += 1
        return list(seen.values())

    def delete_document(self, doc_id: str):
        keep_meta = [m for m in self.meta if m["doc_id"] != doc_id]
        if len(keep_meta) == len(self.meta):
            return False

        model = _get_model()
        with _lock:
            if keep_meta:
                texts      = [m["text"] for m in keep_meta]
                embeddings = model.encode(texts, normalize_embeddings=True)
                new_index  = faiss.IndexFlatIP(EMBED_DIM)
                new_index.add(np.array(embeddings, dtype="float32"))
            else:
                new_index = faiss.IndexFlatIP(EMBED_DIM)

            self.index = new_index
            self.meta  = keep_meta
            self._save()
        return True


def save_uploaded_file(file_storage, original_filename: str):
    """يحفظ الملف المرفوع ويعيد (doc_id, file_path, ext)"""
    ext       = os.path.splitext(original_filename)[1].lower()
    doc_id    = uuid.uuid4().hex[:12]
    safe_name = f"{doc_id}{ext}"
    file_path = os.path.join(UPLOADS_DIR, safe_name)
    file_storage.save(file_path)
    return doc_id, file_path, ext


def build_context_block(results):
    """يبني نص السياق الذي يُرسل للنموذج من نتائج البحث"""
    if not results:
        return None

    lines = ["## معلومات ذات صلة من المستندات المرفوعة:\n"]
    for r in results:
        lines.append(f"[من: {r['source']}]\n{r['text']}\n")
    return "\n".join(lines)
