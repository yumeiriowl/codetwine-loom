"""Index step 4 and search step 2: the BM25 inverted index and scoring.

The tokenizer splits identifiers into their parts as well as into whole words.
"""

import math
import re
from collections import Counter

from codetwine_loom import config
from codetwine_loom.cobol import NAME_JOIN_TEXT
from codetwine_loom.index.store import Index
from codetwine_loom.languages import (LANG_DICT, STOPWORD_BASE_SET, lang_ext,
                                      split_def_key)

# A word: a letter or "_" and the word characters after it
WORD = re.compile(r'[^\W\d]\w*')
# The parts of an identifier: a lower-case run with any capital in front of it, a
# capital run no lower-case letter follows, or a digit run
CAMEL = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+')
# Words joined by hyphens, which is how a COBOL name is written (CUST-ACCT-ID). A match
# starts where a word starts
JOIN_WORD = re.compile(r'(?<!\w)\w+(?:[' + re.escape(NAME_JOIN_TEXT) + r']\w+)+')

# The shortest ASCII word that goes into the index; anything shorter is dropped. A word
# outside ASCII goes in whatever its length
MIN_TOKEN_CHAR = 2


def _is_token(word: str, stop_set: frozenset[str]) -> bool:
    """Say whether a lower-cased word goes into the index."""
    return word not in stop_set and (len(word) >= MIN_TOKEN_CHAR or not word.isascii())


def tokenize(text: str, ext: str | None = None, keep_keyword: bool = False) -> list[str]:
    """Split English text and identifiers into tokens.

    Examples:
        "LLM_API_BASE"          -> ["llm_api_base", "llm", "api", "base"]
        "parse_file(x)"         -> ["parse_file", "parse", "file"]  # the 1-char x is dropped
        "CUST-ACCT-ID"          -> ["cust-acct-id", "cust", "acct", "id"]  # ext None
        "01 CUST-NAME PIC X(30)." -> ["cust-name", "cust", "name"]  # ext "cbl"

    Args:
        text: The text to split.
        ext: The extension of the file the text belongs to, or None. That language's
            keywords and type names are dropped too, and its hyphen-joined words are
            kept whole besides their parts where the language writes names that way
            (hyphen_name). With None, hyphen-joined words are always kept whole besides
            their parts.
        keep_keyword: True keeps the language's keywords and type names.

    Returns:
        The lower-cased tokens. Stopwords, ASCII words shorter than MIN_TOKEN_CHAR and
        hyphen-joined words with no letter in them are dropped.
    """
    stop_set = STOPWORD_BASE_SET
    lang_config = LANG_DICT.get(ext) if ext else None
    if lang_config and not keep_keyword:
        stop_set = stop_set | lang_config.stopword_set
    token_list = []
    if ext is None or (lang_config is not None and lang_config.hyphen_name):
        token_list += [word.lower() for word in JOIN_WORD.findall(text)
                       if any(char.isalpha() for char in word)
                       and _is_token(word.lower(), stop_set)]
    for word in WORD.findall(text):
        lower_word = word.lower()
        if _is_token(lower_word, stop_set):
            token_list.append(lower_word)
        for part in CAMEL.findall(word):
            part = part.lower()
            if part != lower_word and _is_token(part, stop_set):
                token_list.append(part)
    return token_list


def name_token_list(name_list: list[str], ext: str) -> list[str]:
    """Turn definition names into one token each, the whole name.

    Examples:
        (["CUST-ACCT-ID", "WS-A", "FILLER"], "cbl") -> ["cust-acct-id", "ws-a"]

    Args:
        name_list: The names.
        ext: The extension of their file, whose language's keywords are dropped.

    Returns:
        Each name lower-cased, those that are stopwords or ASCII words shorter than
        MIN_TOKEN_CHAR left out.
    """
    lang_config = LANG_DICT.get(ext)
    stop_set = STOPWORD_BASE_SET | (lang_config.stopword_set if lang_config else frozenset())
    return [name.lower() for name in name_list if _is_token(name.lower(), stop_set)]


def build_bm25(doc_dict: dict) -> dict:
    """Build the inverted index (step 4).

    The extension is taken from the file path a definition key starts with and passed to
    tokenize. Only the code half is split with that language's keywords taken out; the
    prose half keeps the words that are keywords in some language. The names a document
    holds are one token each (name_token_list()) and do not count toward its length. The
    path words are split as code, and each of their tokens counts config.PATH_WEIGHT
    times, in the occurrences and in the length; with a weight of 0 they are left out.

    Args:
        doc_dict: {definition key: (code text, prose text[, held names[, path
            words]])}, the result of bm25_doc_dict() or file_bm25_doc_dict().

    Returns:
        {"postings": {term: [[definition key, occurrences], ...]},
         "len": {definition key: token count},
         "avg_len": float}
    """
    posting_dict, len_dict = {}, {}
    for key, doc_tuple in doc_dict.items():
        code_text, prose_text = doc_tuple[:2]
        ext = lang_ext(split_def_key(key)[0])
        path_token_list = tokenize(doc_tuple[3], ext) \
            if len(doc_tuple) > 3 and config.PATH_WEIGHT > 0 else []
        token_list = tokenize(code_text, ext) + tokenize(prose_text, ext, keep_keyword=True)
        len_dict[key] = config.PATH_WEIGHT * len(path_token_list) + len(token_list)
        token_list += name_token_list(list(doc_tuple[2:3] and doc_tuple[2]), ext)
        count_dict: Counter = Counter()
        for token in path_token_list:
            count_dict[token] += config.PATH_WEIGHT
        count_dict.update(token_list)
        for token, count in count_dict.items():
            posting_dict.setdefault(token, []).append([key, count])
    return {"postings": posting_dict, "len": len_dict,
            "avg_len": sum(len_dict.values()) / max(1, len(len_dict))}


def bm25_rank(query: str, index: Index, kind: str) -> list[dict]:
    """Score one index's documents against a query, best first (search step 2).

    The postings come from index.posting_list(), and the number of documents and their
    average length from index.doc_count() and index.avg_len().

    Args:
        query: The search terms as a space-separated string.
        index: The Index to read the postings from.
        kind: "def" or "file".

    Returns:
        [{"key": str, "score": float, "raw": float, "hit_terms": int}, ...], score
        descending. With kind "def" and config.INCLUDE_CONSTANT off, the definitions
        that are constants (is_constant) are not among them.

        score ... divided by the total of the set, which then sums to 1
        raw ... the BM25 score before that division, divided by log(1 + n_docs) and by
            hit_terms (at least 1)
        hit_terms ... how many distinct query terms that tell documents apart hit that
            definition. A term more than half the documents carry is not counted
    """
    # idf's n_docs is how many documents this search covers. avg_len normalises document
    # length and stays as it was at index time. It is 0 when no document that takes
    # part holds a word of its own
    doc_count = index.doc_count(kind)
    avg_len = index.avg_len(kind)
    score_dict, hit_term_dict = {}, {}
    for term in set(tokenize(query)):
        posting_list = index.posting_list(kind, term)
        if not posting_list:
            continue
        df = len(posting_list)
        weight = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        # A term in more than half the documents weighs less than log(2); it is scored
        # like any other and counted as a hit by nobody
        tells_apart = df * 2 <= doc_count
        for doc_id, term_count, doc_len in posting_list:
            # A document's length against the average one; 1 where the average is 0
            length_rate = doc_len / avg_len if avg_len else 1.0
            tf = term_count * (config.BM25_K1 + 1) / (
                term_count + config.BM25_K1 * (1 - config.BM25_B
                                               + config.BM25_B * length_rate))
            score_dict[doc_id] = score_dict.get(doc_id, 0.0) + weight * tf
            hit_term_dict[doc_id] = hit_term_dict.get(doc_id, 0) + tells_apart
    raw_dict = dict(score_dict)
    # What raw is divided by; n_docs is 0 only when nothing was searched
    scale = math.log(1 + doc_count) or 1.0
    total = sum(score_dict.values())
    if total > 0:
        score_dict = {doc_id: score / total for doc_id, score in score_dict.items()}
    key_dict = index.key_dict(kind, list(score_dict))
    return sorted(({"key": key_dict[doc_id], "score": score,
                    "raw": raw_dict[doc_id] / scale / max(1, hit_term_dict[doc_id]),
                    "hit_terms": hit_term_dict[doc_id]}
                   for doc_id, score in score_dict.items()),
                  key=lambda rank_dict: -rank_dict["score"])
