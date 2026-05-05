"""Compact tool-call format codec for the fine-tuned FunctionGemma model.

The model was fine-tuned to emit tool calls in a compact form (~10-20 tokens
per call, ~5x faster decode than canonical JSON on the 2-core A55):

    <tool_2>("red")<end>           -> set_led_color(color="red")
    <tool_5>("beep")<end>          -> play_buzzer(pattern="beep")
    <tool_10>("hello there")<end>  -> respond(message="hello there")

Argument order is positional: required params first, then optional params in
schema declaration order. Trailing nulls are trimmed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOLS_PATH = Path(__file__).resolve().parent.parent / "tools.json"

TOKEN_TO_NAME: dict[str, str | None] = {
    "<tool_0>": "turn_on_lights",
    "<tool_1>": "turn_off_lights",
    "<tool_2>": "set_led_color",
    "<tool_3>": "blink_lights",
    "<tool_4>": "set_neopixel_pattern",
    "<tool_5>": "play_buzzer",
    "<tool_6>": "set_alarm",
    "<tool_7>": "cancel_alarm",
    "<tool_8>": "list_alarms",
    "<tool_9>": "get_system_status",
    "<tool_10>": "respond",
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


def _parse_one(token: str, body: str) -> ToolCall | None:
    name = TOKEN_TO_NAME.get(token)
    if name is None:
        return None
    body = body.strip()
    if name == "respond":
        try:
            message = json.loads(body) if body else ""
        except json.JSONDecodeError:
            message = body.strip().strip('"')
        return ToolCall(name="respond", arguments={"message": message})

    order = _PARAM_ORDER.get(name, [])
    raw_args = _split_top_level(body) if body else []
    arguments: dict[str, Any] = {}
    for k, raw in zip(order, raw_args):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw.strip().strip('"')
        if parsed is None:
            continue
        arguments[k] = parsed
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
