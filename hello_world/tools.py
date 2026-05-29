"""Safe evaluation of arithmetic expressions.

`eval()` would execute arbitrary code, so we walk the AST ourselves and only
allow numbers and a fixed set of arithmetic operators. Anything else raises.
"""

import ast
import operator
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from consts import TOOL_NAME

# Whitelist: only these AST node operators are allowed.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression string, e.g. "(3 + 5) * 2" -> 16.

    Only numbers and + - * / // % ** (plus parentheses) are allowed.
    Raises ValueError on anything else, SyntaxError on malformed input,
    and ZeroDivisionError / OverflowError on bad math.
    """

    def _eval(node: ast.AST) -> float:
        # A literal number. Reject bool (it's a subclass of int) and everything
        # non-numeric (strings, etc.).
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant: {node.value!r}")
            return node.value
        # Binary op: a OP b
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        # Unary op: -a / +a
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return _eval(ast.parse(expression, mode="eval").body)


def get_current_time(tool_input: dict) -> dict:
    timezone: str = tool_input.get("timezone", "Asia/Shanghai")
    print(f"Getting current time for timezone: {timezone}")
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z")
        return {
            "content": f"The current time in {timezone} is {now}.",
            "is_error": False,
        }
    except ZoneInfoNotFoundError:
        return {
            "content": (f"Unknown timezone: {timezone!r}. Use an IANA name like 'Asia/Tokyo' or 'UTC'."),
            "is_error": True,
        }


def calculator(tool_input: dict) -> dict:
    expression: str | None = tool_input.get("expression")
    print(f"Calculating expression: {expression}")

    if expression is None:
        return {"content": "No expression provided for calculation.", "is_error": True}
    try:
        result = safe_eval(expression)
        return {"content": f"The result of {expression} is {result}.", "is_error": False}
    except Exception as e:
        return {"content": f"Error calculating expression: {e}", "is_error": True}


def time_offset(tool_input: dict) -> dict:
    base_time: str = tool_input.get("base_time")
    offset_seconds: int = tool_input.get("offset_seconds")
    try:
        base_dt = datetime.fromisoformat(base_time)
        result = base_dt + timedelta(seconds=offset_seconds)
        return {"content": result.isoformat(), "is_error": False}
    except Exception as e:
        return {"content": f"Invalid base_time format: {e}", "is_error": True}


tool_map = {
    TOOL_NAME.GET_CURRENT_TIME: get_current_time,
    TOOL_NAME.CALCULATE: calculator,
    TOOL_NAME.TIME_OFFSET: time_offset,
}
