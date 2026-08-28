"""Lightweight persistent memory for Railway.

No SentenceTransformer, PyTorch, FAISS, GPU, or Hugging Face inference.
Uses PostgreSQL/SQLite persistence from database.py plus Arabic normalization,
fuzzy similarity, token overlap, and character n-grams to recognize typos and
common rephrasings without loading a large ML model.
"""

import os
import re
import threading
import unicodedata
from difflib import SequenceMatcher

import database

DEFAULT_THRESHOLD = float(os.environ.get("MEMORY_THRESHOLD", "0.64"))
EXACT_THRESHOLD = float(os.environ.get("MEMORY_EXACT_THRESHOLD", "0.88"))
FUZZY_THRESHOLD = float(os.environ.get("MEMORY_FUZZY_THRESHOLD", "0.78"))

_lock = threading.RLock()

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_NON_WORD = re.compile(r"[^\w\s\u0600-\u06FF]+", re.UNICODE)
_SPACES = re.compile(r"\s+")

_REPLACEMENTS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
})

# Words that carry little meaning when comparing Arabic questions.
_STOPWORDS = {
    "ما", "ماذا", "ماهي", "ماهي", "ماهو", "ما هي", "ما هو",
    "هل", "من", "متى", "اين", "اين", "كيف", "لماذا", "ليش", "لماذا",
    "كم", "اي", "ايش", "وش", "شو", "عن", "هو", "هي", "هذا", "هذه",
    "the", "a", "an", "what", "which", "who", "where", "when", "why", "how",
}


def normalize_text(text):
    text = str(text or "").strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS.sub("", text).translate(_REPLACEMENTS)
    text = text.replace("ـ", "")
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _tokens(text):
    n = normalize_text(text)
    return [x for x in n.split() if x not in _STOPWORDS and len(x) > 1]


def _char_similarity(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b), autojunk=False).ratio()


def _token_similarity(a, b):
    aa, bb = set(_tokens(a)), set(_tokens(b))
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def _partial_token_similarity(a, b):
    aa, bb = _tokens(a), _tokens(b)
    if not aa or not bb:
        return 0.0
    matched = 0
    for x in aa:
        best = max(SequenceMatcher(None, x, y, autojunk=False).ratio() for y in bb)
        if best >= 0.72:
            matched += 1
    return matched / max(len(aa), len(bb))


def _compact(text):
    return normalize_text(text).replace(" ", "")


def _score(question, stored):
    nq, ns = normalize_text(question), normalize_text(stored)
    if not nq or not ns:
        return 0.0, 0.0, 0.0, 0.0
    if nq == ns or _compact(nq) == _compact(ns):
        return 1.0, 1.0, 1.0, 1.0

    fuzzy = _char_similarity(nq, ns)
    tokens = _token_similarity(nq, ns)
    partial = _partial_token_similarity(nq, ns)

    # Character similarity handles typos; token/partial scores handle
    # reordered and lightly rephrased questions.
    combined = 0.50 * fuzzy + 0.30 * tokens + 0.20 * partial
    return combined, fuzzy, tokens, partial


class LearnedMemory:
    """Persistent memory with lightweight typo/rephrase matching."""

    def __init__(self, threshold=None):
        self.threshold = float(threshold if threshold is not None else DEFAULT_THRESHOLD)
        print(
            f"[MEMORY] threshold={self.threshold} "
            f"fuzzy={FUZZY_THRESHOLD} exact={EXACT_THRESHOLD}"
        )

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

            best = None
            for item in rows:
                stored = item.get("question", "")
                score, fuzzy, tokens, partial = _score(question, stored)
                candidate = {
                    "id": item["id"],
                    "question": stored,
                    "answer": item["answer"],
                    "source": item.get("source", "unknown"),
                    "source_urls": item.get("source_urls", []),
                    "score": float(score),
                    "fuzzy_score": float(fuzzy),
                    "token_score": float(tokens),
                    "partial_token_score": float(partial),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate

            if not best:
                return None

            # Exact/near-exact typo matches are accepted more aggressively.
            accepted = (
                best["fuzzy_score"] >= EXACT_THRESHOLD
                or (best["fuzzy_score"] >= FUZZY_THRESHOLD and best["score"] >= threshold)
                or best["score"] >= threshold
            )

            if accepted:
                print(
                    f"[MEMORY HIT] score={best['score']:.4f} "
                    f"fuzzy={best['fuzzy_score']:.4f} "
                    f"tokens={best['token_score']:.4f} "
                    f"question={best['question']}"
                )
                return best

            print(f"[MEMORY MISS] best_score={best['score']:.4f}")
            return None
        except Exception as e:
            print(f"[MEMORY SEARCH ERROR] {e}")
            return None

    def save(self, question, answer, source="gemini", source_urls=None):
        question = str(question or "").strip()
        answer = str(answer or "").strip()
        if not question or not answer:
            raise ValueError("question و answer مطلوبان")
        source_urls = source_urls or []

        with _lock:
            # Existing schema requires an embedding column. We intentionally
            # store an empty vector because this lightweight memory does not
            # use embeddings at all.
            existing = self.search(question, threshold=min(self.threshold, 0.80))
            if existing and existing["score"] >= 0.80:
                database.update_learned_knowledge(
                    existing["id"], question, answer, [], source, source_urls
                )
                print(f"[MEMORY UPDATE] id={existing['id']} source={source}")
                return {
                    "id": existing["id"], "question": question, "answer": answer,
                    "source": source, "source_urls": source_urls, "updated": True,
                }

            memory_id = database.save_learned_knowledge(
                question, answer, [], source, source_urls
            )
            print(f"[MEMORY SAVE] id={memory_id} source={source}")
            return {
                "id": memory_id, "question": question, "answer": answer,
                "source": source, "source_urls": source_urls, "updated": False,
            }

    def count(self):
        return database.count_learned_knowledge()

    def list_items(self):
        rows = database.get_all_learned_knowledge()
        return [
            {
                "id": x["id"], "question": x["question"], "answer": x["answer"],
                "source": x.get("source"), "source_urls": x.get("source_urls", []),
                "created_at": x.get("created_at"), "updated_at": x.get("updated_at"),
            }
            for x in reversed(rows)
        ]

    def delete(self, memory_id):
        return database.delete_learned_knowledge(memory_id)
