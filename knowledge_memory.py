import os
import re
import threading
from difflib import SequenceMatcher

import numpy as np

import database
import rag

DEFAULT_THRESHOLD = float(os.environ.get("MEMORY_THRESHOLD", "0.64"))
_lock = threading.Lock()


def _normalize(text):
    """تطبيع بسيط للعربية والإنجليزية لمقاومة الأخطاء الإملائية الشكلية."""
    text = str(text or "").strip().lower()
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[؟?!.,،؛:;\-_/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _text_similarity(a, b):
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


class LearnedMemory:
    """ذاكرة معرفة دائمة وخفيفة لا تحتاج نموذج Transformer أو GPU."""

    def __init__(self, threshold=None):
        self.threshold = float(threshold if threshold is not None else DEFAULT_THRESHOLD)
        print(f"[MEMORY] threshold={self.threshold}")

    def search(self, question, threshold=None):
        question = str(question or "").strip()
        if not question:
            return None

        threshold = self.threshold if threshold is None else float(threshold)

        try:
            rows = database.get_all_learned_knowledge()
            if not rows:
                print("[MEMORY MISS] empty")
                return None

            # نعيد إنشاء المتجهات من الأسئلة نفسها حتى تعمل الذاكرة أيضًا
            # مع سجلات قديمة كانت محفوظة بمتجهات 384-dim.
            normalized_question = _normalize(question)
            questions = [item.get("question", "") for item in rows]
            vectors = rag._get_embeddings([normalized_question] + [_normalize(q) for q in questions])
            query_vector = vectors[0]

            best = None
            for idx, item in enumerate(rows, start=1):
                vector_score = _similarity(query_vector, vectors[idx])
                fuzzy_score = _text_similarity(question, item.get("question", ""))

                # المزج يجعل المطابقة مقاومة للأخطاء الإملائية وإعادة الصياغة
                # البسيطة، مع بقاء التشابه الدلالي الحرفي هو العامل الأكبر.
                score = (0.78 * vector_score) + (0.22 * fuzzy_score)

                candidate = {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "source": item.get("source", "unknown"),
                    "source_urls": item.get("source_urls", []),
                    "score": float(score),
                    "vector_score": float(vector_score),
                    "fuzzy_score": float(fuzzy_score),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate

            if best and best["score"] >= threshold:
                print(
                    f"[MEMORY HIT] score={best['score']:.4f} "
                    f"vector={best['vector_score']:.4f} "
                    f"fuzzy={best['fuzzy_score']:.4f} "
                    f"question={best['question']}"
                )
                return best

            print(
                "[MEMORY MISS] "
                + (f"best_score={best['score']:.4f}" if best else "empty")
            )
            return None

        except Exception as e:
            print(f"[MEMORY SEARCH ERROR] {e}")
            return None

    def save(self, question, answer, source="user", source_urls=None):
        question = str(question or "").strip()
        answer = str(answer or "").strip()
        if not question or not answer:
            raise ValueError("question و answer مطلوبان")

        source_urls = source_urls or []

        with _lock:
            # إذا كانت نفس المعرفة موجودة، نحدّثها بدل إنشاء نسخة ثانية.
            existing = self.search(question, threshold=min(self.threshold, 0.80))
            if existing and existing["score"] >= 0.80:
                embedding = rag._get_embeddings([_normalize(question)])[0].tolist()
                database.update_learned_knowledge(
                    existing["id"], question, answer, embedding, source, source_urls
                )
                print(f"[MEMORY UPDATE] id={existing['id']}")
                return {**existing, "question": question, "answer": answer,
                        "source": source, "source_urls": source_urls, "updated": True}

            embedding = rag._get_embeddings([_normalize(question)])[0].tolist()
            memory_id = database.save_learned_knowledge(
                question, answer, embedding, source, source_urls
            )
            print(f"[MEMORY SAVE] id={memory_id} source={source}")
            return {
                "id": memory_id,
                "question": question,
                "answer": answer,
                "source": source,
                "source_urls": source_urls,
                "updated": False,
            }

    def count(self):
        return database.count_learned_knowledge()

    def list_items(self):
        rows = database.get_all_learned_knowledge()
        return [
            {
                "id": x["id"],
                "question": x["question"],
                "answer": x["answer"],
                "source": x.get("source"),
                "source_urls": x.get("source_urls", []),
                "created_at": x.get("created_at"),
                "updated_at": x.get("updated_at"),
            }
            for x in reversed(rows)
        ]

    def delete(self, memory_id):
        return database.delete_learned_knowledge(memory_id)
