"""Reading COBOL source lines.

The code of a line is read by codetwine's code_text_list(): columns 8-72 of a
fixed-format line (a full-width character counting as two columns), the whole line in
free format, nothing for a comment or a directive line. Which format the lines are in is
decided from the lines given.
"""

from codetwine.parsers.cobol_format import code_text_list

# The characters a COBOL word is made of besides \w: the hyphen, in ASCII and full width
NAME_JOIN_TEXT = "-－"


def line_key_list(line_list: list[str]) -> list[str]:
    """Return the form each line is compared in.

    The sequence and identification areas, the case and the amount of space between the
    words are left out.

    Examples:
        ["000800 Main-Para." + " " * 55 + "MAIN0008"] -> ["MAIN-PARA."]
        ["       MAIN-PARA."]                          -> ["MAIN-PARA."]

    Args:
        line_list: Lines of one file, or a run of them, without newlines.

    Returns:
        One string per line: the words of its code in upper case, joined by one space.
    """
    return [" ".join(text.split()).upper() for text in code_text_list(line_list)]
