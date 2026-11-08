"""Cut texts to a limit and split them into lines.

The module also holds the checks of a text a tool was handed and the commands a message
hands its reader.
"""

import json
import re
import shlex

# The marker appended to a text whose end was cut off, in three forms: the text itself,
# its shape with N for the count, and the regex that finds it at the end of a text
CUT_NOTE = "\n... ({omitted} chars omitted)"
CUT_NOTE_SHAPE = "... (N chars omitted)"
CUT_NOTE_RE = re.compile(r"\.\.\. \(\d+ chars omitted\)$")

# The marker put in front of a text whose head was dropped to keep its tail
CUT_HEAD_NOTE = "... ({omitted} chars omitted)\n"

# The most digits the count of a marker is read with. A longer run of digits is no
# marker
_MAX_COUNT_DIGIT = 15

# The two markers as a cut left them, whatever the count (group 1) and whatever the one
# white-space character between the marker and the text became
_CUT_NOTE_ANY_RE = re.compile(
    r"\s?\.\.\. \((\d{1,%d}) chars omitted\)" % _MAX_COUNT_DIGIT)
_CUT_HEAD_NOTE_ANY_RE = re.compile(
    r"\.\.\. \((\d{1,%d}) chars omitted\)\s?" % _MAX_COUNT_DIGIT)


def _omit_count(drop_text: str, mark_re: re.Pattern) -> int | None:
    """Count the characters a cut leaves out, reading the markers of earlier cuts.

    Args:
        drop_text: The part of the text the cut does not keep.
        mark_re: _CUT_NOTE_ANY_RE or _CUT_HEAD_NOTE_ANY_RE.

    Returns:
        The length of drop_text, each marker in it counted as the number it states
        and not as its own length. None when drop_text is one marker and nothing else,
        which is a text already cut.
    """
    if mark_re.fullmatch(drop_text):
        return None
    return len(drop_text) + sum(int(mark[1]) - len(mark[0])
                                for mark in mark_re.finditer(drop_text))


def cut(text: str, max_char: int) -> str:
    """Cut a string down to the limit and mark that it was cut.

    A marker an earlier cut left is read wherever it stands: one that would be cut in
    two is left out whole, and the count the new marker states is the number of
    characters of the first text that are left out. White space behind a marker is
    text as any other.

    Args:
        text: The string before cutting.
        max_char: How many characters to keep. 0 or less keeps the whole string.

    Returns:
        The string unchanged when it fits or there is no cap, and when it is what a cut
        to this limit or a lower one left: at most max_char characters followed by the
        marker and nothing behind it. Otherwise its first max_char characters followed
        by the "... (N chars omitted)" marker.
    """
    if max_char <= 0 or len(text) <= max_char:
        return text
    keep_end = max_char
    for mark in _CUT_NOTE_ANY_RE.finditer(text):
        if mark.start() <= max_char < mark.end():
            keep_end = mark.start()
    omit_count = _omit_count(text[keep_end:], _CUT_NOTE_ANY_RE)
    if omit_count is None:
        return text
    return text[:keep_end] + CUT_NOTE.format(omitted=omit_count)


def cut_tail(text: str, max_char: int) -> str:
    """Keep the tail of a string up to the limit and mark that its head was dropped.

    A marker an earlier cut left is read as cut() reads one.

    Args:
        text: The string before cutting.
        max_char: How many characters of the tail to keep. 0 or less keeps the whole
            string.

    Returns:
        The string unchanged when it fits or there is no cap, and when it is what a cut
        to this limit or a lower one left: the marker, with nothing in front of it,
        followed by at most max_char characters. Otherwise the "... (N chars omitted)"
        marker on a line of its own followed by its last max_char characters.
    """
    if max_char <= 0 or len(text) <= max_char:
        return text
    keep_start = len(text) - max_char
    for mark in _CUT_HEAD_NOTE_ANY_RE.finditer(text):
        if mark.start() < len(text) - max_char <= mark.end():
            keep_start = mark.end()
    omit_count = _omit_count(text[:keep_start], _CUT_HEAD_NOTE_ANY_RE)
    if omit_count is None:
        return text
    return CUT_HEAD_NOTE.format(omitted=omit_count) + text[keep_start:]


def cut_by(text: str, drop_count: int) -> str:
    """Shorten a text by a number of characters, the marker of the cut counted in.

    Args:
        text: The string before cutting.
        drop_count: How many characters shorter the text is to be at least. 0 or less
            leaves it as it is.

    Returns:
        The text cut at its end and marked as cut() marks it, at least drop_count
        characters shorter than it was with the marker counted as the characters it
        takes written as JSON. The marker alone, without its line break, when no
        character of the text can stay beside it, and "" when the marker cannot
        either.
    """
    if drop_count <= 0:
        return text
    note = CUT_NOTE.format(omitted=len(text))
    keep_count = len(text) - drop_count - len(json.dumps(note)) + 2
    if keep_count > 0:
        cut_text = cut(text, keep_count)
        if len(cut_text) < len(text):
            return cut_text
    note = note.strip()
    return note if len(note) <= len(text) - drop_count else ""


def cut_entry_text(entry_dict: dict, over_count: int,
                   keep_key_tuple: tuple[str, ...] = ()) -> int:
    """Cut the texts one entry holds by a number of characters in all, the longest first.

    Args:
        entry_dict: The entry, whose values that are strings are its texts. It is
            changed in place, and a text is cut at its end and marked (cut_by()).
        over_count: How many characters are to go.
        keep_key_tuple: The keys whose text is left as it is.

    Returns:
        How many of the over_count characters are still there once every text is
        empty: 0 when the texts held that many.
    """
    key_list = sorted((key for key, value in entry_dict.items()
                       if isinstance(value, str) and key not in keep_key_tuple),
                      key=lambda key: len(entry_dict[key]), reverse=True)
    for key in key_list:
        if over_count <= 0:
            break
        text = entry_dict[key]
        entry_dict[key] = cut_by(text, over_count)
        over_count -= len(text) - len(entry_dict[key])
    return max(0, over_count)


def json_over_count(value: object, budget: int) -> int:
    """Return how many characters a value written as JSON is over a budget.

    The value is written as json.dumps(value, ensure_ascii=False) writes it.

    Args:
        value: The value.
        budget: How many characters it may take. 0 or less is no budget.

    Returns:
        The number of characters, 0 for a value within the budget and for no budget.
    """
    if budget <= 0:
        return 0
    return max(0, len(json.dumps(value, ensure_ascii=False)) - budget)


def limit_text(text: str, limit: int) -> str:
    """Fill in a phrase that states a limit, or leave it out when there is no limit.

    A limit of 0 or less is no limit.

    Examples:
        ("Keep it under {limit} characters.", 6000) -> "Keep it under 6000 characters."
        ("Keep it under {limit} characters.", 0)    -> ""

    Args:
        text: The phrase, with "{limit}" where the number goes.
        limit: The limit the phrase states.

    Returns:
        The phrase with the limit filled in, or "" when the limit is 0 or less.
    """
    return text.format(limit=limit) if limit > 0 else ""


# The most characters of names one log line lists (name_list_text)
LOG_NAME_MAX_CHAR = 20_000


def name_list_text(name_list: list[str], max_char: int = LOG_NAME_MAX_CHAR) -> str:
    """Join names with ", " up to a number of characters, with a count for the rest.

    The first name is always written.

    Examples:
        (["a.py", "b.py"], 100)          -> "a.py, b.py"
        (["a.py", "b.py", "c.py"], 10)   -> "a.py, b.py and 1 more"

    Args:
        name_list: The names, in the order they are written.
        max_char: The most characters the joined names take.

    Returns:
        The joined names, followed by " and N more" when some are left out. "" for no
        names.
    """
    shown_list: list[str] = []
    char_count = 0
    for name in name_list:
        char_count += len(name) + (2 if shown_list else 0)
        if shown_list and char_count > max_char:
            break
        shown_list.append(name)
    rest_count = len(name_list) - len(shown_list)
    head = ", ".join(shown_list)
    return f"{head} and {rest_count} more" if rest_count else head


def one_line(text: str) -> str:
    """Fold every run of whitespace, newlines included, into one space.

    Args:
        text: The text to flatten.

    Returns:
        The text as one line, without leading or trailing spaces.
    """
    return " ".join(text.split())


def command_line(*word_tuple: str) -> str:
    """Return a command as a line a POSIX shell reads back as the same words.

    Examples:
        command_line("git", "-C", "/a  b", "diff", "x$(y)")
            -> "git -C '/a  b' diff 'x$(y)'"

    Args:
        *word_tuple: The command's words: its name, its flags and their values.

    Returns:
        The words, each as shlex.quote() writes it, with one space between them. A
        word that holds a newline keeps it inside its quotes.
    """
    return shlex.join(word_tuple)


class CommandError(RuntimeError):
    """An error whose message ends with a command its reader runs by hand.

    The message is why followed by command.

    Attributes:
        why: What went wrong, up to where the command follows.
        command: The command, as command_line() wrote it.
    """

    def __init__(self, why: str, command: str) -> None:
        """Hold the two parts of the message.

        Args:
            why: What went wrong, up to where the command follows.
            command: The command, as command_line() wrote it.
        """
        super().__init__(f"{why} {command}")
        self.why = why
        self.command = command


def error_part(error: BaseException) -> tuple[str, str]:
    """Return an error's message apart from the command it ends with.

    Args:
        error: What was raised.

    Returns:
        (why, command) of a CommandError, and (the message, "") of any other error.
    """
    if isinstance(error, CommandError):
        return error.why, error.command
    return str(error), ""


def with_command(why: str, command: str) -> str:
    """Return a message followed by the command it ends with, when it has one.

    Args:
        why: The message, as error_part() returns it.
        command: The command, or "".

    Returns:
        why and command with one space between them, or the one that is not empty.
    """
    return " ".join(text for text in (why, command) if text)


# The characters a terminal acts on: the C0 controls but tab and newline, DEL and the
# C1 controls
_TERM_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def term_text(text: str) -> str:
    """Return a text with every character a terminal acts on written as "\\xNN".

    Tab and newline are kept. Every other C0 control, DEL and every C1 control is
    written as a backslash, "x" and its two hex digits.

    Examples:
        "a\\x1b[2Jb\\n"  -> "a\\\\x1b[2Jb\\n"
        "a\\rb\\x07"     -> "a\\\\x0db\\\\x07"

    Args:
        text: The text as it was made.

    Returns:
        The text a terminal shows character by character.
    """
    return _TERM_CONTROL_RE.sub(lambda match: f"\\x{ord(match[0]):02x}", text)


# One line with its ending, the ending one of "\n", "\r\n" and a lone "\r", or the last
# line of a text that does not end in one
_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")


def line_list_of(text: str, keepends: bool = False) -> list[str]:
    """Split a text into its lines at "\\n", "\\r\\n" and a lone "\\r" alone.

    A form feed, a vertical tab, U+0085, U+2028 and the other characters str.splitlines
    also breaks at stay inside their line.

    Examples:
        ("a\\n\\f\\nb\\n")        -> ["a", "\\f", "b"]
        ("a\\n\\fb", True)       -> ["a\\n", "\\fb"]
        ("a\\u2028b\\n")         -> ["a\\u2028b"]
        ("")                   -> []

    Args:
        text: The text to split.
        keepends: True keeps each line's ending on it.

    Returns:
        The lines. A text that ends in a line break has no empty line after it.
    """
    line_list = _LINE_RE.findall(text)
    if keepends:
        return line_list
    return [line.rstrip("\r\n") for line in line_list]


# A character that is half of a surrogate pair, and what stands in a text where one was
_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")
_NO_CHARACTER = "\ufffd"


def no_lone_surrogate(node: object) -> object:
    """Return a value without the halves of surrogate pairs in its strings.

    A string holding one cannot be encoded as UTF-8.

    Args:
        node: A string, or a list or dict as json.loads() returns them.

    Returns:
        The value with every such character, in a string or in a key at any depth,
        replaced by U+FFFD. A value that is none of the three is returned as it is.
    """
    if isinstance(node, str):
        return _LONE_SURROGATE_RE.sub(_NO_CHARACTER, node)
    if isinstance(node, list):
        return [no_lone_surrogate(item) for item in node]
    if isinstance(node, dict):
        return {no_lone_surrogate(key): no_lone_surrogate(item)
                for key, item in node.items()}
    return node


def text_error(**text_dict: object) -> str:
    """Say which of the arguments a tool was handed as text is not a string.

    Args:
        **text_dict: {the argument's name: what was passed}.

    Returns:
        "<name> must be a string, not <type>" for the first that is not one, "" when
        all are.
    """
    for name, value in text_dict.items():
        if not isinstance(value, str):
            return f"{name} must be a string, not {type(value).__name__}"
    return ""


def whole_number_error(name: str, value: object) -> str:
    """Say why a value a tool was handed as a line number or a count is not one.

    Args:
        name: The argument's name, for the message.
        value: What was passed.

    Returns:
        "<name> must be a whole number, not <value>" for anything but an int (a bool
        is not one), "" for an int.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{name} must be a whole number, not {value!r}"
    return ""
