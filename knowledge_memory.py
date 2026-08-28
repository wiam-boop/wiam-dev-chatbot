"""
Improved Long-Term Knowledge Memory

Features:
- PostgreSQL/SQLite persistence through database.py
- Local multilingual SentenceTransformer embeddings
- Arabic text normalization
- Diacritics / punctuation / whitespace normalization
- Common Arabic spelling normalization
- Character-level fuzzy similarity for typos
- Semantic similarity for rephrased questions
- Combined confidence scoring
- Prevents Web fallback when a reliable memory match exists
- Does not call Hugging Face Inference API
"""

import os
import re
import threading
import unicodedata
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer

import database


# =========================================================
# CONFIG
# =========================================================

EMBED_MODEL_NAME = os.environ.get(
    "MEMORY_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
).strip()

DEFAULT_THRESHOLD = float(
    os.environ.get("MEMORY_THRESHOLD", "0.64")
)

# Very high confidence = accept immediately.
EXACT_FUZZY_THRESHOLD = float(
    os.environ.get("MEMORY_EXACT_FUZZY_THRESHOLD", "0.90")
)

# Semantic-only threshold for genuinely rephrased questions.
SEMANTIC_THRESHOLD = float(
    os.environ.get("MEMORY_SEMANTIC_THRESHOLD", "0.74")
)

# Combined score threshold.
COMBINED_THRESHOLD = float(
    os.environ.get("MEMORY_COMBINED_THRESHOLD", "0.68")
)

# For small/medium personal knowledge bases this is fine.
# Later, for tens of thousands of memories, pgvector is recommended.
MAX_MEMORIES_IN_CACHE = int(
    os.environ.get("MEMORY_CACHE_LIMIT", "10000")
)


# =========================================================
# MODEL
# =========================================================

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                print("[MEMORY] Loading local embedding model...")

                _model = SentenceTransformer(
                    EMBED_MODEL_NAME
                )

                print("[MEMORY] Local memory embedding model loaded.")

    return _model


# =========================================================
# TEXT NORMALIZATION
# =========================================================

_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

_PUNCTUATION = re.compile(
    r"[^\w\s\u0600-\u06FF]"
)

_SPACES = re.compile(r"\s+")

# These substitutions intentionally normalize common writing variants
# without changing the meaning of the sentence.
_ARABIC_REPLACEMENTS = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
}


def normalize_text(text):
    """
    Normalize Arabic and general text for matching.

    Examples:
        "ماهي عاصمة الهند؟"
        "ماهي عاصمى الهند"
        "ما هي عاصمة الهند"
    become much closer for fuzzy matching.
    """

    text = str(text or "").strip().lower()

    if not text:
        return ""

    # Unicode normalization.
    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # Remove Arabic tashkeel/diacritics.
    text = _ARABIC_DIACRITICS.sub(
        "",
        text
    )

    # Normalize Arabic letter variants.
    for old, new in _ARABIC_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Normalize common Arabic elongation.
    text = text.replace("ـ", "")

    # Remove punctuation.
    text = _PUNCTUATION.sub(
        " ",
        text
    )

    # Collapse whitespace.
    text = _SPACES.sub(
        " ",
        text
    ).strip()

    return text


def compact_text(text):
    """
    Remove spaces after normalization.

    This is useful for small spelling differences such as:
    "عاصمة" vs "عاصم ة"
    """
    return normalize_text(text).replace(" ", "")


# =========================================================
# FUZZY MATCHING
# =========================================================

def _sequence_similarity(a, b):
    if not a or not b:
        return 0.0

    return float(
        SequenceMatcher(
            None,
            a,
            b,
            autojunk=False
        ).ratio()
    )


def _token_similarity(a, b):
    """
    Compare normalized words while tolerating small spelling errors.
    """

    a_tokens = normalize_text(a).split()
    b_tokens = normalize_text(b).split()

    if not a_tokens or not b_tokens:
        return 0.0

    # For each token in A, find the closest token in B.
    scores_a = []

    for token_a in a_tokens:
        best = max(
            (
                _sequence_similarity(
                    token_a,
                    token_b
                )
                for token_b in b_tokens
            ),
            default=0.0
        )

        scores_a.append(best)

    # Same in reverse so extra unrelated words are penalized.
    scores_b = []

    for token_b in b_tokens:
        best = max(
            (
                _sequence_similarity(
                    token_b,
                    token_a
                )
                for token_a in a_tokens
            ),
            default=0.0
        )

        scores_b.append(best)

    return float(
        (
            sum(scores_a) / len(scores_a)
            +
            sum(scores_b) / len(scores_b)
        ) / 2.0
    )


def fuzzy_similarity(question_a, question_b):
    """
    Combined typo-aware lexical similarity.
    """

    normalized_a = normalize_text(question_a)
    normalized_b = normalize_text(question_b)

    if not normalized_a or not normalized_b:
        return 0.0

    if normalized_a == normalized_b:
        return 1.0

    compact_a = normalized_a.replace(" ", "")
    compact_b = normalized_b.replace(" ", "")

    sequence_score = _sequence_similarity(
        normalized_a,
        normalized_b
    )

    compact_score = _sequence_similarity(
        compact_a,
        compact_b
    )

    token_score = _token_similarity(
        normalized_a,
        normalized_b
    )

    return float(
        max(
            sequence_score,
            compact_score,
            token_score
        )
    )


# =========================================================
# EMBEDDINGS
# =========================================================

def _get_embedding(text):
    text = str(text or "").strip()

    if not text:
        raise ValueError(
            "النص فارغ"
        )

    model = _get_model()

    vector = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    return np.asarray(
        vector,
        dtype=np.float32
    )


def _cosine_similarity(a, b):
    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    if a.size == 0 or b.size == 0:
        return 0.0

    if a.shape != b.shape:
        return 0.0

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / (a_norm * b_norm)
    )


# =========================================================
# MEMORY CLASS
# =========================================================

class LearnedMemory:

    def __init__(self, threshold=None):

        self.threshold = (
            float(threshold)
            if threshold is not None
            else DEFAULT_THRESHOLD
        )

        self._lock = threading.Lock()

        # In-memory cache. Database remains the source of truth.
        self._cache = None

        print(
            f"[MEMORY] threshold={self.threshold}"
        )

    # =====================================================
    # CACHE
    # =====================================================

    def _load_cache(self):

        if self._cache is not None:
            return self._cache

        with self._lock:

            if self._cache is None:

                memories = (
                    database
                    .get_all_learned_knowledge()
                )

                if (
                    MAX_MEMORIES_IN_CACHE > 0
                    and len(memories) > MAX_MEMORIES_IN_CACHE
                ):
                    memories = memories[
                        -MAX_MEMORIES_IN_CACHE:
                    ]

                self._cache = memories

                print(
                    "[MEMORY] Cache loaded: "
                    f"{len(memories)} memories"
                )

        return self._cache

    def refresh(self):

        with self._lock:
            self._cache = None

        return self._load_cache()

    # =====================================================
    # SEARCH
    # =====================================================

    def search(self, question, threshold=None):

        question = str(
            question or ""
        ).strip()

        if not question:
            return None

        threshold = (
            self.threshold
            if threshold is None
            else float(threshold)
        )

        try:

            normalized_question = normalize_text(
                question
            )

            compact_question = (
                normalized_question
                .replace(" ", "")
            )

            query_embedding = _get_embedding(
                question
            )

            memories = self._load_cache()

            if not memories:

                print("[MEMORY MISS] empty")

                return None

            best = None

            for item in memories:

                stored_question = item.get(
                    "question",
                    ""
                )

                normalized_stored = normalize_text(
                    stored_question
                )

                compact_stored = (
                    normalized_stored
                    .replace(" ", "")
                )

                # -----------------------------------------
                # 1. Exact normalized match
                # -----------------------------------------

                if (
                    normalized_question
                    == normalized_stored
                    and normalized_question
                ):

                    score = 1.0
                    fuzzy_score = 1.0
                    semantic_score = 1.0
                    match_type = "exact"

                else:

                    # -------------------------------------
                    # 2. Fuzzy / typo matching
                    # -------------------------------------

                    fuzzy_score = fuzzy_similarity(
                        question,
                        stored_question
                    )

                    # -------------------------------------
                    # 3. Semantic matching
                    # -------------------------------------

                    semantic_score = _cosine_similarity(
                        query_embedding,
                        item.get("embedding", [])
                    )

                    # -------------------------------------
                    # 4. Combined score
                    # -------------------------------------

                    # Fuzzy is intentionally given more weight
                    # for short Arabic questions with typos.
                    score = (
                        0.60 * fuzzy_score
                        +
                        0.40 * semantic_score
                    )

                    # A very strong fuzzy match should not be
                    # dragged down by a weak embedding score.
                    if fuzzy_score >= EXACT_FUZZY_THRESHOLD:
                        score = max(
                            score,
                            fuzzy_score
                        )

                    # A strong semantic match is enough for
                    # genuine rephrasing.
                    if semantic_score >= SEMANTIC_THRESHOLD:
                        score = max(
                            score,
                            semantic_score
                        )

                    match_type = "semantic+fuzzy"

                candidate = {
                    "id": item["id"],
                    "question": stored_question,
                    "answer": item["answer"],
                    "source": item.get(
                        "source",
                        "unknown"
                    ),
                    "source_urls": item.get(
                        "source_urls",
                        []
                    ),
                    "score": float(score),
                    "fuzzy_score": float(
                        fuzzy_score
                    ),
                    "semantic_score": float(
                        semantic_score
                    ),
                    "match_type": match_type,
                    "created_at": item.get(
                        "created_at"
                    ),
                    "updated_at": item.get(
                        "updated_at"
                    ),
                }

                if (
                    best is None
                    or candidate["score"]
                    > best["score"]
                ):
                    best = candidate

            if not best:
                print("[MEMORY MISS] no candidates")
                return None

            # =================================================
            # ACCEPTANCE RULES
            # =================================================

            accepted = False

            # Exact normalized match.
            if best["match_type"] == "exact":
                accepted = True

            # Strong typo match.
            elif (
                best["fuzzy_score"]
                >= EXACT_FUZZY_THRESHOLD
            ):
                accepted = True

            # Strong semantic paraphrase.
            elif (
                best["semantic_score"]
                >= SEMANTIC_THRESHOLD
                and best["score"]
                >= threshold
            ):
                accepted = True

            # Good combined match.
            elif (
                best["score"]
                >= COMBINED_THRESHOLD
            ):
                accepted = True

            if accepted:

                print(
                    "[MEMORY HIT] "
                    f"score={best['score']:.4f} "
                    f"fuzzy={best['fuzzy_score']:.4f} "
                    f"semantic={best['semantic_score']:.4f} "
                    f"type={best['match_type']} "
                    f"question={best['question']}"
                )

                return best

            print(
                "[MEMORY MISS] "
                f"best_score={best['score']:.4f} "
                f"fuzzy={best['fuzzy_score']:.4f} "
                f"semantic={best['semantic_score']:.4f}"
            )

            return None

        except Exception as e:

            print(
                f"[MEMORY SEARCH ERROR] {e}"
            )

            return None

    # =====================================================
    # SAVE
    # =====================================================

    def save(
        self,
        question,
        answer,
        source="user",
        source_urls=None
    ):

        question = str(
            question or ""
        ).strip()

        answer = str(
            answer or ""
        ).strip()

        if not question:
            raise ValueError(
                "question لا يمكن أن يكون فارغًا"
            )

        if not answer:
            raise ValueError(
                "answer لا يمكن أن يكون فارغًا"
            )

        source_urls = source_urls or []

        embedding = _get_embedding(
            question
        )

        with self._lock:

            # Check existing memory with a very strict threshold
            # to update duplicates instead of creating them.
            existing = self.search(
                question,
                threshold=0.90
            )

            if existing and (
                existing["fuzzy_score"] >= 0.90
                or existing["semantic_score"] >= 0.90
            ):

                updated = (
                    database
                    .update_learned_knowledge(
                        memory_id=existing["id"],
                        question=question,
                        answer=answer,
                        embedding=embedding.tolist(),
                        source=source,
                        source_urls=source_urls,
                    )
                )

                if updated:

                    self.refresh()

                    print(
                        "[MEMORY UPDATE] "
                        f"id={existing['id']}"
                    )

                    return {
                        "id": existing["id"],
                        "question": question,
                        "answer": answer,
                        "source": source,
                        "source_urls": source_urls,
                        "updated": True,
                    }

            memory_id = (
                database
                .save_learned_knowledge(
                    question=question,
                    answer=answer,
                    embedding=embedding.tolist(),
                    source=source,
                    source_urls=source_urls,
                )
            )

            # Update cache without querying on next request.
            self._cache = None

            print(
                "[MEMORY SAVE] "
                f"id={memory_id} "
                f"source={source}"
            )

            return {
                "id": memory_id,
                "question": question,
                "answer": answer,
                "source": source,
                "source_urls": source_urls,
                "updated": False,
            }

    # =====================================================
    # COUNT
    # =====================================================

    def count(self):
        return database.count_learned_knowledge()

    # =====================================================
    # LIST
    # =====================================================

    def list_items(self):

        memories = (
            database
            .get_all_learned_knowledge()
        )

        return [
            {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "source": item.get("source"),
                "source_urls": item.get(
                    "source_urls",
                    []
                ),
                "created_at": item.get(
                    "created_at"
                ),
                "updated_at": item.get(
                    "updated_at"
                ),
            }
            for item in reversed(memories)
        ]

    # =====================================================
    # DELETE
    # =====================================================

    def delete(self, memory_id):

        deleted = (
            database
            .delete_learned_knowledge(
                memory_id
            )
        )

        if deleted:

            self.refresh()

            print(
                "[MEMORY DELETE] "
                f"id={memory_id}"
            )

        return deleted
