"""Lightweight long-term knowledge memory.

- Uses the same embedding model as rag.py (no second Transformer model).
- Handles Arabic normalization, typos and rephrased questions.
- Stores durable knowledge in the existing PostgreSQL/SQLite database.
- Does not call the Web; main.py decides when Web is allowed.
"""

import os
import re
import threading
from difflib import SequenceMatcher

import numpy as np

import database
import rag

DEFAULT_THRESHOLD = float(os.environ.get("MEMORY_THRESHOLD", "0.64"))
FUZZY_THRESHOLD = float(os.environ.get("MEMORY_FUZZY_THRESHOLD", "0.86"))
SEMANTIC_THRESHOLD = float(os.environ.get("MEMORY_SEMANTIC_THRESHOLD", "0.78"))
COMBINED_THRESHOLD = float(os.environ.get("MEMORY_COMBINED_THRESHOLD", "0.74"))

_lock = threading.Lock()


def normalize_text(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("گ", "ك").replace("ڨ", "ق")
    text = re.sub(r"[؟?!.,،؛:;\-_\/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_similarity(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def token_similarity(a, b):
    aa = normalize_text(a).split()
    bb = normalize_text(b).split()
    if not aa or not bb:
        return 0.0
    # Character-level similarity plus token overlap makes short Arabic typo
    # questions much more robust than embeddings alone.
    sa, sb = set(aa), set(bb)
    overlap = len(sa & sb) / max(1, len(sa | sb))
    return 0.65 * text_similarity(a, b) + 0.35 * overlap


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class LearnedMemory:
    """Persistent semantic memory with typo/paraphrase matching."""

    def __init__(self, threshold=None):
        self.threshold = float(threshold if threshold is not None else DEFAULT_THRESHOLD)
        print(
            f"[MEMORY] threshold={self.threshold} "
            f"fuzzy={FUZZY_THRESHOLD} semantic={SEMANTIC_THRESHOLD}"
        )

    def _embed(self, texts):
        # Reuse rag.py's single embedding model. This is important on Railway.
        return rag._get_embeddings([normalize_text(x) for x in texts])

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

            query_vector = self._embed([question])[0]
            best = None

            for item in rows:
                stored_question = item.get("question", "")
                semantic = cosine(query_vector, item.get("embedding", []))
                fuzzy = text_similarity(question, stored_question)
                token = token_similarity(question, stored_question)

                # For short Arabic questions, fuzzy similarity is intentionally
                # important. For genuine paraphrases, semantic similarity wins.
                combined = 0.45 * semantic + 0.40 * fuzzy + 0.15 * token
                score = max(combined, fuzzy if fuzzy >= 0.92 else 0.0)

                candidate = dict(item)
                candidate.update({
                    "score": float(score),
                    "semantic_score": float(semantic),
                    "fuzzy_score": float(fuzzy),
                    "token_score": float(token),
                })

                if best is None or candidate["score"] > best["score"]:
                    best = candidate

            if not best:
                print("[MEMORY MISS] no candidates")
                return None

            # Acceptance rules:
            # 1) almost identical wording / typo
            # 2) strong semantic paraphrase + reasonable combined score
            # 3) combined score high enough
            accepted = (
                best["fuzzy_score"] >= FUZZY_THRESHOLD
                or (
                    best["semantic_score"] >= SEMANTIC_THRESHOLD
                    and best["score"] >= threshold
                )
                or best["score"] >= COMBINED_THRESHOLD
            )

            if accepted:
                print(
                    f"[MEMORY HIT] score={best['score']:.4f} "
                    f"semantic={best['semantic_score']:.4f} "
                    f"fuzzy={best['fuzzy_score']:.4f} "
                    f"question={best['question']}"
                )
                return best

            print(
                f"[MEMORY MISS] best_score={best['score']:.4f} "
                f"semantic={best['semantic_score']:.4f} "
                f"fuzzy={best['fuzzy_score']:.4f}"
            )
            return None

        except Exception as exc:
            print(f"[MEMORY SEARCH ERROR] {exc}")
            return None

    def save(self, question, answer, source="web", source_urls=None, metadata=None):
        question = str(question or "").strip()
        answer = str(answer or "").strip()
        source = str(source or "unknown").strip()

        if not question or not answer:
            raise ValueError("question و answer مطلوبان")

        # Do not save obvious failure messages as knowledge.
        bad_markers = [
            "حدث خطأ مؤقت",
            "حدث خطأ غير متوقع",
            "الخدمة مشغولة",
            "تعذر الاتصال",
            "please try again",
            "temporary error",
        ]
        if any(x.lower() in answer.lower() for x in bad_markers):
            raise ValueError("لن يتم حفظ رسالة خطأ في الذاكرة")

        embedding = self._embed([question])[0].tolist()
        source_urls = list(source_urls or [])
        if not source_urls and isinstance(metadata, dict):
            source_urls = list(metadata.get("web_sources") or [])

        with _lock:
            rows = database.get_all_learned_knowledge()
            normalized = normalize_text(question)
            existing_id = None
            best_existing = None

            for item in rows:
                stored = item.get("question", "")
                if normalize_text(stored) == normalized:
                    existing_id = item["id"]
                    break
                sim = text_similarity(question, stored)
                if best_existing is None or sim > best_existing[0]:
                    best_existing = (sim, item)

            # Update near-duplicate knowledge instead of creating dozens of
            # entries for spelling variants of the same fact.
            if existing_id is None and best_existing and best_existing[0] >= 0.93:
                existing_id = best_existing[1]["id"]

            if existing_id is not None:
                database.update_learned_knowledge(
                    existing_id,
                    question,
                    answer,
                    embedding,
                    source,
                    source_urls,
                )
                print(f"[MEMORY UPDATE] id={existing_id} source={source}")
                return {
                    "id": existing_id,
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "source_urls": source_urls,
                    "updated": True,
                }

            memory_id = database.save_learned_knowledge(
                question,
                answer,
                embedding,
                source,
                source_urls,
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

    # Compatibility with older main.py versions.
    list_all = list_items

    def delete(self, memory_id):
        return database.delete_learned_knowledge(memory_id)

    def clear(self):
        rows = database.get_all_learned_knowledge()
        for item in rows:
            database.delete_learned_knowledge(item["id"])
