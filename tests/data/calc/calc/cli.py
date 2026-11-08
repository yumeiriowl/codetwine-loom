"""Command line: calc <op> <a> <b>."""

import sys

from calc.basic import add
from calc.scale import mul

OP_DICT = {"add": add, "mul": mul}


def main(argv: list[str] | None = None) -> int:
    """Print the result of one operation and return 0."""
    op, a, b = (argv or sys.argv[1:])[:3]
    print(OP_DICT[op](float(a), float(b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
