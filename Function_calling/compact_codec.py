"""Compact tool-call format codec for the fine-tuned FunctionGemma model.

Two formats are accepted, decided per-arg at parse time:

  Positional (v9, SmartPanel-style):
      <tool_0>("red","on")<end>     -> set_status_led(led="red", state="on")
      <tool_3>("beep")<end>         -> play_buzzer(pattern="beep")
      <tool_7>("hello there")<end>  -> respond(message="hello there")

  Named (v10, Mercedes-Benz Octopus v2 style, arXiv 2501.02342):
      <tool_0>(color="red", state="on")<end>      -> set_lights(...)
      <tool_0>(effect="rainbow", speed="slow")    -> set_lights(...)
      <tool_5>(message="hello")<end>              -> respond(message="hello")

Positional order is required params first, then optional params in schema
declaration order; trailing nulls trimmed. Named args may appear in any order
and absent args are simply not emitted. Mixed positional + named in a single
call is accepted (positional args fill slots left-to-right, named args
override their named slot).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOLS_PATH = Path(__file__).resolve().parent / "tools.json"

TOKEN_TO_NAME: dict[str, str | None] = {
    "<tool_0>": "set_lights",
    "<tool_1>": "play_buzzer",
    "<tool_2>": "set_alarm",
    "<tool_3>": "cancel_alarm",
    "<tool_4>": "get_system_status",
    "<tool_5>": "respond",
    "<tool_none>": None,
}

_COMPACT_RE = re.compile(r"<tool_(\d+|none)>\(([^)]*)\)(?:<end>)?")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


def _load_param_order() -> dict[str, list[str]]:
    schema = json.loads(TOOLS_PATH.read_text())
    out: dict[str, list[str]] = {}
    for tool in schema["tools"]:
        fn = tool["function"]
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        optional = [k for k in props.keys() if k not in required]
        out[fn["name"]] = required + optional
    return out


_PARAM_ORDER = _load_param_order()


def _split_top_level(s: str) -> list[str]:
    """Split comma-separated JSON literals, respecting quotes and nested braces."""
    out: list[str] = []
    depth = 0
    buf: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if ch == "\\":
            buf.append(ch)
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            buf.append(ch)
            continue
        if in_str:
            buf.append(ch)
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return out


def _find_named_split(raw: str) -> int | None:
    """Return the index of the first '=' that's outside any quoted string,
    or None if there's no top-level '='. Used to detect named-arg syntax."""
    in_str = False
    esc = False
    for i, ch in enumerate(raw):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str and ch == "=":
            return i
    return None


def _coerce_value(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip('"')


def _parse_one(token: str, body: str) -> ToolCall | None:
    name = TOKEN_TO_NAME.get(token)
    if name is None:
        return None
    body = body.strip()

    # respond with no '=' in the body: treat the entire body as the message
    # so quoted messages containing commas (e.g. "Hi, there!") don't get
    # mis-split. v10 named form (message="...") falls through to the general
    # parser below.
    if name == "respond" and _find_named_split(body) is None:
        try:
            message = json.loads(body) if body else ""
        except json.JSONDecodeError:
            message = body.strip('"')
        return ToolCall(name="respond", arguments={"message": message})

    order = _PARAM_ORDER.get(name, [])
    raw_args = _split_top_level(body) if body else []
    arguments: dict[str, Any] = {}
    positional_idx = 0
    for raw in raw_args:
        eq = _find_named_split(raw)
        if eq is not None:
            arg_name = raw[:eq].strip()
            value = _coerce_value(raw[eq + 1:])
            if value is None:
                continue
            arguments[arg_name] = value
        else:
            value = _coerce_value(raw)
            if value is None:
                positional_idx += 1
                continue
            if positional_idx < len(order):
                arguments[order[positional_idx]] = value
            positional_idx += 1
    return ToolCall(name=name, arguments=arguments)


def parse_compact(text: str) -> list[ToolCall]:
    """Parse zero or more compact-format tool calls from raw model output."""
    calls: list[ToolCall] = []
    for match in _COMPACT_RE.finditer(text):
        token = f"<tool_{match.group(1)}>"
        body = match.group(2)
        call = _parse_one(token, body)
        if call is not None:
            calls.append(call)
    return calls
