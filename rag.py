"""
محرك RAG — يستخدم HuggingFace للـ embeddings و Groq (Qwen Vision) لفهم الصور
"""
import base64
import os
import json
import uuid
import threading

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

from pypdf import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract

# ─── مسار التخزين ───
IS_RAILWAY = os.environ.get("RAILWAY_ENVIRONMENT") is not None

# STORAGE_DIR قابل للضبط عبر متغير بيئة STORAGE_PATH.
# على Railway: أنشئي Volume من تبويب Variables/Volumes في الخدمة،
# اربطيه بمسار مثل /data، وضعي STORAGE_PATH=/data في Environment Variables.
# بدون هذا الضبط، سيُستعمل /tmp/storage كحل احتياطي فقط —
# وهو غير دائم ويُمسح مع كل إعادة نشر/تشغيل على Railway.
_env_storage_path = os.environ.get("STORAGE_PATH", "").strip()

if _env_storage_path:
    STORAGE_DIR = _env_storage_path
elif IS_RAILWAY:
    STORAGE_DIR = "/tmp/storage"
    print(
        "[تحذير] STORAGE_PATH غير مضبوط على Railway. "
        "سيتم استخدام /tmp/storage وهو غير دائم "
        "(ستُفقد الملفات والفهرس مع كل إعادة نشر). "
        "أضيفي Volume دائم واضبطي STORAGE_PATH لتفادي فقدان البيانات."
    )
else:
    STORAGE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "storage"
    )

UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
INDEX_PATH  = os.path.join(STORAGE_DIR, "kb.index")
META_PATH   = os.path.join(STORAGE_DIR, "kb_meta.json")

os.makedirs(UPLOADS_DIR, exist_ok=True)

# ─── نموذج embedding محلي (بلا API خارجي) ───
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM = 384
_embed_model = None
_embed_model_lock = threading.Lock()

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _embed_model_lock:
            if _embed_model is None:
                print("[RAG] Loading local embedding model...")
                _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
                print("[RAG] Local embedding model loaded.")
    return _embed_model

# ─── عميل Groq (مخصص فقط لفهم الصور عبر نموذج Vision) ───
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_groq_client = Groq(api_key=GROQ_API_KEY)
VISION_MODEL = "qwen/qwen3.6-27b"   # نموذج Groq متعدد الوسائط (نص + صورة) — متاح في الخطة المجانية

CHUNK_SIZE    = 700
CHUNK_OVERLAP = 120
TOP_K         = 4
_lock         = threading.Lock()


def _get_embeddings(texts: list) -> np.ndarray:
    """ينشئ embeddings محليًا بدون HuggingFace Inference API."""
    model = _get_embed_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return np.asarray(embeddings, dtype="float32")

# ─── استخراج النص ───

def extract_text(file_path: str, ext: str) -> str:
    ext = ext.lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
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
    return "\n".join(
        (p.extract_text() or "") for p in reader.pages
        if (p.extract_text() or "").strip()
    )


def _extract_docx(file_path: str) -> str:
    doc   = DocxDocument(file_path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text for cell in row.cells)
            if row_text.strip():
                parts.append(row_text)
    return "\n".join(parts)


def _extract_image(file_path: str) -> str:
    """
    يحاول أولاً قراءة أي نص مكتوب داخل الصورة عبر OCR.
    إذا لم يجد نصاً كافياً (صورة بدون كتابة: منتج، مكان، رسم، صورة شخص...)
    يستخدم نموذج رؤية (Vision) عبر Groq لوصف محتوى الصورة نصياً،
    ليصبح هذا الوصف قابلاً للفهرسة والبحث ضمن قاعدة المعرفة.
    """
    image = Image.open(file_path)

    # الخطوة 1: محاولة OCR أولاً (أسرع وأرخص، ومناسب للمستندات الممسوحة)
    try:
        ocr_text = pytesseract.image_to_string(image, lang="ara+eng")
    except Exception:
        try:
            ocr_text = pytesseract.image_to_string(image)
        except Exception:
            ocr_text = ""

    ocr_text = ocr_text.strip()

    # لو استخرجنا نصاً معقولاً (أكثر من 20 حرف مثلاً) نكتفي به
    if len(ocr_text) > 20:
        return ocr_text

    # الخطوة 2: لا يوجد نص كافٍ → استخدم نموذج الرؤية لوصف الصورة
    vision_description = _describe_image_with_vision(file_path)

    # ندمج الاثنين احتياطاً (في حال وجد OCR كلمات قليلة مفيدة)
    combined = "\n".join(filter(None, [ocr_text, vision_description]))
    return combined.strip()


def _describe_image_with_vision(file_path: str) -> str:
    """يستخدم نموذج رؤية عبر Groq API لوصف محتوى الصورة نصياً بالتفصيل."""
    if not GROQ_API_KEY:
        return "[لم يتم ضبط GROQ_API_KEY، تعذر تحليل الصورة]"

    try:
        with open(file_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(file_path)[1].lower().replace(".", "")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext

        completion = _groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "صف هذه الصورة بالتفصيل باللغة العربية: "
                                "ما الذي تظهره؟ الأشخاص، الأشياء، النصوص إن وجدت، "
                                "الألوان، السياق العام. اجعل الوصف غنياً بالمعلومات "
                                "بحيث يمكن الإجابة على أسئلة عن محتوى الصورة اعتماداً عليه فقط."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime};base64,{b64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.3,
            max_completion_tokens=800,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        # في حال فشل الاتصال بنموذج الرؤية، لا نكسر الرفع بالكامل
        return f"[تعذر توليد وصف للصورة تلقائياً: {e}]"


# ─── تقسيم النص ───

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


# ─── قاعدة المعرفة ───

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
        embeddings = _get_embeddings(chunks)
        with _lock:
            self.index.add(embeddings)
            for i, chunk in enumerate(chunks):
                self.meta.append({
                    "doc_id": doc_id, "source": source_name,
                    "chunk_no": i, "text": chunk,
                })
            self._save()
        return len(chunks)

    def search(self, query: str, top_k=TOP_K):
        if self.index.ntotal == 0:
            return []
        q_emb           = _get_embeddings([query])
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
                seen[doc_id] = {"doc_id": doc_id, "source": item["source"], "chunks": 0}
            seen[doc_id]["chunks"] += 1
        return list(seen.values())

    def delete_document(self, doc_id: str):
        keep_meta = [m for m in self.meta if m["doc_id"] != doc_id]
        if len(keep_meta) == len(self.meta):
            return False
        with _lock:
            if keep_meta:
                new_index = faiss.IndexFlatIP(EMBED_DIM)
                new_index.add(_get_embeddings([m["text"] for m in keep_meta]))
            else:
                new_index = faiss.IndexFlatIP(EMBED_DIM)
            self.index = new_index
            self.meta  = keep_meta
            self._save()
        return True


def save_uploaded_file(file_storage, original_filename: str):
    ext       = os.path.splitext(original_filename)[1].lower()
    doc_id    = uuid.uuid4().hex[:12]
    file_path = os.path.join(UPLOADS_DIR, f"{doc_id}{ext}")
    file_storage.save(file_path)
    return doc_id, file_path, ext


def build_context_block(results):
    if not results:
        return None
    lines = ["## معلومات ذات صلة من المستندات المرفوعة:\n"]
    for r in results:
        lines.append(f"[من: {r['source']}]\n{r['text']}\n")
    return "\n".join(lines)
