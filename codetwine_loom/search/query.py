"""Search step 1: rewrite the request into English search terms.

A request in another language is rewritten into English words. When there is a
repository overview (OVERVIEW_TEXT), the rewrite is handed it and runs for an English
request too.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from codetwine_loom import config
from codetwine_loom.cobol import NAME_JOIN_TEXT

logger = logging.getLogger(__name__)

# The repository's requirements, as the overview says them. "" when there is no overview
OVERVIEW_TEXT = ""

# What step 1 asks the model. {words} is QUERY_EXPAND_WORD_RANGE and {overview_block}
# is _overview_block()
_EXPAND_PROMPT = """Rewrite this software change request as English \
search keywords for a codebase search. The request may be in any language.

Output a single line of space-separated English words: the content words \
plus likely identifier names, synonyms, and related technical terms that \
would appear in the source code or its design docs. {words} words. No punctuation, \
no explanation.
{overview_block}
Request:
{request}
"""

# The block put in the prompt when there is an overview
_OVERVIEW_BLOCK = """
What this repository does (its requirements). Use it to tell which of its features the \
request is about:
{overview_text}
"""


# What separates the parts of a path
PATH_MARK_TUPLE = ("/", "\\")
# A word of a path or of a name: letters, digits and "_", with "-" between them
NAME_WORD = re.compile(r"\w+(?:-+\w+)*")
# The same in bare text, where a name is written in ASCII
_BARE_WORD = re.compile(r"[A-Za-z0-9_]+(?:-+[A-Za-z0-9_]+)*")
# A word in backquotes, the backquote itself, a run of backquotes, and a line of a
# request
_QUOTE_WORD = re.compile(r"`([^`\s]+)`")
_QUOTE_MARK = re.compile("`")
_QUOTE_RUN = re.compile("`+")
_LINE = re.compile(r"[^\r\n]+")
# A run of bare text a name may be written in: printable ASCII with neither a space
# nor a backquote in it
_BARE_TEXT = re.compile(r"[!-_a-~]+")
# What starts the address of a URL, the characters its scheme is written with, and the
# rest of it from that mark on
_URL_MARK = "://"
_SCHEME_MARK_TEXT = "+.-"
_URL_REST = re.compile(r"(?:(?![`\"'<>)\]])[!-~])*")
# A search term in bare text: a run of ASCII letters, digits, "_", ".", ":", "-" and
# path separators that starts with a letter, a digit, a "_" or a "."
_BARE_TERM = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.:/\\-]*")
# A search term in a word in backquotes: a run of letters and digits (outside ASCII
# too), "_", ".", ":", the hyphens of NAME_JOIN_TEXT and path separators that starts
# with a letter, a digit, a "_" or a "."
_QUOTE_TERM = re.compile(r"[\w.][\w.:/\\" + re.escape(NAME_JOIN_TEXT) + "]*")
# What is taken off the end of a search term
_NAME_EDGE = ".:" + NAME_JOIN_TEXT + "".join(PATH_MARK_TUPLE)
# What tells a bare name from a word: a slash, a dot followed by a letter, an
# underscore, a hyphen or a lower-to-upper case change
_NAME_MARK = re.compile(r"/|\.[A-Za-z]|_|-|[a-z][A-Z]")


@dataclass(frozen=True)
class WordRun:
    """A stretch of a request, read as words and what stands between them.

    Attributes:
        text: The stretch as it is written.
        word_list: Its words, in order.
        gap_list: What stands before each word and, last, behind the last word. One
            longer than word_list.
        in_quote: True for a word in backquotes, False for bare text.
        after_text: True when a letter or a digit outside ASCII stands right before
            the stretch.
    """

    text: str
    word_list: list[str]
    gap_list: list[str]
    in_quote: bool
    after_text: bool


def _url_span_list(text: str) -> list[tuple[int, int]]:
    """Return where the URLs of a text are.

    Examples:
        "see(https://x.y/a.py),b.py" -> [(4, 20)]

    Args:
        text: A text with no space in it.

    Returns:
        (start, end) of each URL, in order: from the first character of the scheme
        before _URL_MARK to where _URL_REST ends.
    """
    span_list = []
    position = 0
    while (mark := text.find(_URL_MARK, position)) >= 0:
        start = mark
        while start > position and text[start - 1].isascii() \
                and (text[start - 1].isalnum() or text[start - 1] in _SCHEME_MARK_TEXT):
            start -= 1
        position = _URL_REST.match(text, mark).end()
        span_list.append((start, position))
    return span_list


def _text_run_list(text: str, in_quote: bool, after_text: bool) -> list[WordRun]:
    """Read a text with no space in it as word runs, its URLs left out.

    Args:
        text: A run of _BARE_TEXT, or the inside of a word in backquotes.
        in_quote: Which of the two it is. The words of the first are ASCII.
        after_text: Whether a letter or a digit outside ASCII stands right before text.

    Returns:
        One WordRun for each stretch between the URLs that holds a word, in order.
    """
    word_re = NAME_WORD if in_quote else _BARE_WORD
    run_list = []
    position = 0
    for start, end in _url_span_list(text) + [(len(text), len(text))]:
        piece = text[position:start]
        word_list, gap_list, gap_start = [], [], 0
        for word in word_re.finditer(piece):
            gap_list.append(piece[gap_start:word.start()])
            word_list.append(word.group(0))
            gap_start = word.end()
        if word_list:
            run_list.append(WordRun(piece, word_list, gap_list + [piece[gap_start:]],
                                    in_quote, after_text and position == 0))
        position = end
    return run_list


def _bare_run_list(request: str, start: int, end: int) -> list[WordRun]:
    """Read the bare text between two places of a request as word runs.

    Args:
        request: The change request as the user wrote it.
        start: Where the bare text starts in request.
        end: Where it ends.

    Returns:
        The word runs of each run of _BARE_TEXT (_text_run_list()), in order.
    """
    run_list = []
    for bare_match in _BARE_TEXT.finditer(request, start, end):
        before = request[bare_match.start() - 1] if bare_match.start() else " "
        run_list += _text_run_list(bare_match.group(0), False,
                                   not before.isascii() and before.isalnum())
    return run_list


def _run_pair_list(request: str, run_list: list[tuple[int, int]]
                   ) -> tuple[list[tuple[int, int]], bool]:
    """Pair the runs of backquotes of a line and return the words the pairs hold.

    Each run is one mark. The first run opens. A run as long as the one that opened, or
    longer, closes it: what stands between the two is a word in backquotes when it is
    one word with no backquote in it. The backquotes a longer run has left over open
    again. A shorter run is part of what stands between the two, and when that is no
    word, the shorter runs in it are paired among themselves the same way; the ones that
    leave a run open hold no word.

    A run of the line that no later run is long enough to close opens nothing: the
    words paired in front of it stand, and the run after it opens.

    Args:
        request: The text the runs are in.
        run_list: [(where a run starts, how many backquotes it has), ...], in order.

    Returns:
        ((start, end) of each word in backquotes, from the backquote next to the word
        on its left to the one next to it on its right, in order; whether a run of
        the line is left with no run to close it).
    """
    # The longest run behind each run of the line; 0 behind the last
    longest_after_list = [0] * (len(run_list) + 1)
    for index in range(len(run_list) - 1, -1, -1):
        longest_after_list[index] = max(run_list[index][1], longest_after_list[index + 1])
    span_list: list[tuple[int, int]] = []
    is_left_open = False
    # One entry for each pairing going on, that of the line first and then that of the
    # runs inside each open pair: [how many backquotes are open, where the run that
    # opened ends, how many words span_list held when the pairing began]
    level_list = [[0, 0, 0]]
    for index, (start, length) in enumerate(run_list):
        # A run that closes a pair ends the pairing of the runs inside that pair
        while len(level_list) > 1 and level_list[-2][0] <= length:
            _end_pair_level(level_list, span_list)
        if level_list[-1][0] > length:
            level_list.append([0, 0, len(span_list)])
        level = level_list[-1]
        rest_length = length - level[0]
        if level[0] and _QUOTE_WORD.fullmatch(request, level[1], start + 1):
            span_list.append((level[1], start + 1))
        if len(level_list) == 1 and longest_after_list[index + 1] < rest_length:
            is_left_open, rest_length = True, 0
        level[0], level[1] = rest_length, start + length - 1
    while level_list:
        _end_pair_level(level_list, span_list)
    return sorted(span_list), is_left_open


def _end_pair_level(level_list: list[list[int]], span_list: list[tuple[int, int]]) -> None:
    """End the innermost pairing of _run_pair_list().

    Args:
        level_list: The pairings going on. Its last entry is taken off.
        span_list: The words found so far. The ones found since that pairing began,
            in it and inside its pairs, are taken off when it ends with a run open.
    """
    open_length, _, word_count = level_list.pop()
    if open_length:
        del span_list[word_count:]


def _same_run_pair_list(request: str, run_list: list[tuple[int, int]]
                        ) -> list[tuple[int, int]]:
    """Pair the runs of backquotes of a line by length and return the words they hold.

    Each run pairs with the next run of the same length. A run with no later run of its
    own length pairs with none, and the run after it is looked at next. The runs between
    the two of a pair are part of what the pair holds.

    Args:
        request: The text the runs are in.
        run_list: [(where a run starts, how many backquotes it has), ...], in order.

    Returns:
        (start, end) of each pair that holds one word with no backquote in it, as
        _run_pair_list() returns them, in order.
    """
    # The index of the next run of the same length behind each run; None for none
    next_same_list: list[int | None] = [None] * len(run_list)
    index_by_length_dict: dict[int, int] = {}
    for index in range(len(run_list) - 1, -1, -1):
        next_same_list[index] = index_by_length_dict.get(run_list[index][1])
        index_by_length_dict[run_list[index][1]] = index
    span_list = []
    index = 0
    while index < len(run_list):
        close_index = next_same_list[index]
        if close_index is None:
            index += 1
            continue
        open_at = run_list[index][0] + run_list[index][1] - 1
        if _QUOTE_WORD.fullmatch(request, open_at, run_list[close_index][0] + 1):
            span_list.append((open_at, run_list[close_index][0] + 1))
        index = close_index + 1
    return span_list


def _span_list_with(span_list: list[tuple[int, int]],
                    more_list: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Add to one list of spans the spans of another that overlap none of it.

    Args:
        span_list: (start, end) spans in order, none overlapping another.
        more_list: The same of the spans to add.

    Returns:
        The spans of span_list and those of more_list that share no place with a
        span of span_list, in order.
    """
    merge_list = []
    index = 0
    for start, end in more_list:
        while index < len(span_list) and span_list[index][1] <= start:
            merge_list.append(span_list[index])
            index += 1
        if index == len(span_list) or end <= span_list[index][0]:
            merge_list.append((start, end))
    return merge_list + span_list[index:]


def quote_span_list(request: str) -> list[tuple[int, int]]:
    """Return where the words in backquotes of a request are.

    A run of backquotes is one mark, and the marks of a line are paired from its start
    (_run_pair_list): a pair with one word between its two is a word in backquotes. A
    pair with a space or nothing between its two is none, and neither of its marks
    belongs to another pair.

    A line with a run that no run closes is read another way: its backquotes are taken
    two by two from its start, one by one, when their number is even, and it is read
    from its start for _QUOTE_WORD when it is odd. The words its runs pair to
    (_run_pair_list) are words of it as well, where they share no place with a word
    read that way, and so are the words its runs of the same length pair to
    (_same_run_pair_list), where they share no place with either.

    Examples:
        "`make test` fails, so fix `Makefile`"  -> [(26, 36)]
        "``Makefile`` needs a fix"              -> [(1, 11)]
        "`pyproject.toml and `LICENSE`"         -> [(20, 29)]
        "``Makefile`` and ``LICENSE need a fix" -> [(1, 11)]

    Args:
        request: A change request or search terms, as they were written.

    Returns:
        (start, end) of each word in backquotes, the backquote on each side of the
        word included, in order.
    """
    span_list = []
    for line in _LINE.finditer(request):
        run_list = [(run.start(), len(run.group(0)))
                    for run in _QUOTE_RUN.finditer(request, line.start(), line.end())]
        pair_list, is_left_open = _run_pair_list(request, run_list)
        if not is_left_open:
            span_list += pair_list
            continue
        mark_list = [mark.start() for mark in
                     _QUOTE_MARK.finditer(request, line.start(), line.end())]
        if len(mark_list) % 2:
            more_list = [quote.span() for quote in
                         _QUOTE_WORD.finditer(request, line.start(), line.end())]
        else:
            more_list = [(start, end + 1)
                         for start, end in zip(mark_list[::2], mark_list[1::2])
                         if _QUOTE_WORD.fullmatch(request, start, end + 1)]
        span_list += _span_list_with(_span_list_with(more_list, pair_list),
                                     _same_run_pair_list(request, run_list))
    return span_list


def word_run_list(request: str) -> list[WordRun]:
    """Cut a request into the stretches a name may be written in.

    Examples:
        "`a/b.py` or café.py" -> a run of "a/b.py" in backquotes, bare runs of "or",
                                 "caf" and ".py"
        "see https://x.y/z"   -> a bare run of "see"

    Args:
        request: A change request or search terms, as they were written.

    Returns:
        The word runs in the order they are written. A word in backquotes
        (quote_span_list()) is one stretch, whatever its letters; the text outside is a
        stretch for each run of _BARE_TEXT (printable ASCII with neither a space nor a
        backquote in it) that holds a word. A URL ends a stretch and belongs to none.
    """
    run_list = []
    position = 0
    for start, end in quote_span_list(request):
        run_list += _bare_run_list(request, position, start)
        run_list += _text_run_list(request[start + 1:end - 1], True, False)
        position = end
    return run_list + _bare_run_list(request, position, len(request))


def _run_term_list(run: WordRun, is_known_name: Callable[[str], bool]) -> list[str]:
    """List the search terms one word run holds.

    Args:
        run: A stretch of word_run_list().
        is_known_name: Tells whether the index has a file or a definition of a name.

    Returns:
        The terms in the order they are written: each run of _BARE_TERM (_QUOTE_TERM
        in backquotes), its _NAME_EDGE characters taken off the end, that holds a
        letter and _NAME_MARK. The first term of a word in backquotes needs no
        _NAME_MARK. A term that starts a stretch behind a word outside ASCII with a
        character that is neither a letter nor a digit is one only when is_known_name
        says so.
    """
    term_list = []
    need_mark = not run.in_quote
    for match in (_QUOTE_TERM if run.in_quote else _BARE_TERM).finditer(run.text):
        term = match.group(0).rstrip(_NAME_EDGE)
        if not any(char.isalpha() for char in term) \
                or (need_mark and not _NAME_MARK.search(term)):
            continue
        if run.after_text and match.start() == 0 and not term[0].isalnum() \
                and not is_known_name(term):
            continue
        term_list.append(term)
        need_mark = True
    return term_list


def literal_term_list(request: str,
                      is_known_name: Callable[[str], bool] | None = None) -> list[str]:
    """List the paths, file names and identifiers written in a request.

    Examples:
        "add a dependency to `pyproject.toml`" -> ["pyproject.toml"]
        "store/memory.go and loadConfig()"     -> ["store/memory.go", "loadConfig"]
        "fix `Makefile` and `2to3.py`"         -> ["Makefile", "2to3.py"]
        "fix the width of `café_total`"        -> ["café_total"]
        "`/store/memory.go` and `@app.route`"  -> ["store/memory.go", "app.route"]
        "cfg.py#load_config and Store#loadAll" -> ["cfg.py", "load_config", "loadAll"]
        "`src/a.py#L10,b.py(new)`"             -> ["src/a.py", "b.py"]
        "Python 3.10 and https://x.y/doc"      -> []
        "fix café.py"                          -> []

    Args:
        request: The change request as the user wrote it.
        is_known_name: Tells whether the index has a file or a definition of a given
            name. None knows no name.

    Returns:
        The terms of every stretch of word_run_list() (_run_term_list()), in the order
        they are written, each once:

        - a term is a run of letters, digits, "_", ".", ":", "-", "/" and "\\" (in a
          word in backquotes also the full-width hyphen "－"); any other character ends
          it and the next term starts behind it
        - the marks in front of a term and the ".", ":", "-", "－" and separators at its
          end are taken off
        - a URL and a term with no letter are not terms

        In a word in backquotes the letters may be outside ASCII and the first term
        needs no mark of a name. Everywhere else a term has a slash, a dot followed by
        a letter, an underscore, a hyphen or a lower-to-upper case change in it, and
        one that continues a word outside ASCII with a "." or a "_" is a term only
        when is_known_name says so.
    """
    known_dict: dict[str, bool] = {}

    def is_known(term: str) -> bool:
        """Return whether is_known_name, when given, knows term; each term is asked once."""
        if term not in known_dict:
            known_dict[term] = bool(is_known_name and is_known_name(term))
        return known_dict[term]

    term_list = []
    for run in word_run_list(request):
        term_list += _run_term_list(run, is_known)
    return list(dict.fromkeys(term_list))


# An English word in search terms
_ENGLISH_WORD = re.compile(r"[A-Za-z_]")


def has_english_word(text: str) -> bool:
    """Whether search terms hold an English word: an ASCII letter or an underscore.

    Examples:
        "load config retry" -> True
        "3.10 3.11"         -> False
        "- - -"             -> False

    Args:
        text: Search terms, space separated.

    Returns:
        True when the text holds an ASCII letter or an underscore.
    """
    return bool(_ENGLISH_WORD.search(text))


def _with_name_list(term_list: list[str], request: str,
                    is_known_name: Callable[[str], bool] | None) -> list[str]:
    """Append the names written in the request that the terms lack.

    Args:
        term_list: The search terms the model returned, already cut to word_cap().
        request: The change request as the user wrote it.
        is_known_name: Passed to literal_term_list().

    Returns:
        term_list followed by literal_term_list(request, is_known_name) minus the terms
        already in it, compared whole and case-insensitively.
    """
    lower_set = {term.lower() for term in term_list}
    add_list = [term for term in literal_term_list(request, is_known_name)
                if term.lower() not in lower_set]
    if add_list:
        logger.debug("names written in the request are kept: %s", " ".join(add_list))
    return term_list + add_list


# A number of words in QUERY_EXPAND_WORD_RANGE, written in ASCII digits, and the most
# digits one is written with
_MAX_WORD_COUNT_DIGIT = 9
_WORD_COUNT = re.compile(r"(?<!\d)\d{1,%d}(?!\d)" % _MAX_WORD_COUNT_DIGIT, re.ASCII)


def word_cap() -> int:
    """The most words the expansion may return, taken from the range the prompt states.

    Returns:
        The largest number in QUERY_EXPAND_WORD_RANGE, or 0 when it holds no number at
        all. 0 turns the cut off. A run of more than _MAX_WORD_COUNT_DIGIT digits is no
        number, and neither is one of digits that are not ASCII.
    """
    number_list = [int(number) for number in
                   _WORD_COUNT.findall(str(config.QUERY_EXPAND_WORD_RANGE))]
    return max(number_list) if number_list else 0


def need_expand(request: str) -> bool:
    """Say whether step 1 asks the model to rewrite this request.

    Args:
        request: The change request as the user wrote it.

    Returns:
        True when QUERY_EXPAND is on and the request is not ASCII only or there is an
        overview.
    """
    return config.QUERY_EXPAND and (not request.isascii() or bool(OVERVIEW_TEXT))


def _overview_block() -> str:
    """Return the prompt's overview block, or "" when there is no overview."""
    return _OVERVIEW_BLOCK.format(overview_text=OVERVIEW_TEXT) if OVERVIEW_TEXT else ""


def expand_query(request: str, llm: Callable[[str], str] | None,
                 is_known_name: Callable[[str], bool] | None = None) -> tuple[str, bool]:
    """Rewrite a request into a line of English search terms.

    The 3 gates are asked in order: QUERY_EXPAND off, an ASCII-only request while there
    is no overview, then a missing llm. Each returns the request unchanged and none of
    them is a failure; the second and the third log which gate it was.

    A reply with no English word in it (has_english_word()) is a failure like an
    exception is. It is asked for once more before the request is searched with as it
    is; an exception is not.

    Args:
        request: The change request as the user wrote it.
        llm: A callable taking a string and returning a string.
        is_known_name: Passed to literal_term_list().

    Returns:
        (the search terms, whether the search may be trusted). The terms are the model's
        words cut to word_cap(), followed by the names written in the request that the
        words lack (literal_term_list()); they are the request unchanged when nothing
        was asked for or the call failed. The second member is False only when the model
        was asked and raised an exception, or replied twice with no English word.
    """
    if not config.QUERY_EXPAND:
        return request, True
    if not need_expand(request):
        logger.debug("the request is ASCII only, so the expansion is skipped")
        return request, True
    if llm is None:
        logger.warning("no model is available for the expansion; searching with the "
                       "request as it is, which the English index barely matches")
        return request, True
    for is_last in (False, True):
        try:
            term_list = llm(_EXPAND_PROMPT.format(
                request=request, words=config.QUERY_EXPAND_WORD_RANGE,
                overview_block=_overview_block())).strip().split()
        except Exception as e:
            # Any failure, authentication included, is logged with its type and message
            logger.warning("query expansion failed; searching with the request as it "
                           "is: %s: %s", type(e).__name__, e)
            return request, False
        if has_english_word(" ".join(term_list)):
            cap = word_cap()
            if cap and len(term_list) > cap:
                logger.info("the expansion returned %d words; searching with the first "
                            "%d", len(term_list), cap)
                term_list = term_list[:cap]
            return " ".join(_with_name_list(term_list, request, is_known_name)), True
        logger.warning("the query expansion came back with no English word in it%s",
                       "; searching with the request as it is" if is_last
                       else ", so it is asked for once more")
    return request, False
