"""llama-cpp-python runtime for the fine-tuned FunctionGemma 270M model.

Builds the same prompt shape Synaptics' ``agentic/function_gemma_astra.py``
uses (raw Gemma turn format with ``<start_function_declaration>`` blocks),
runs llama-cpp inference on the 2-core A55 CPU, and parses the compact
tool-call format the model was fine-tuned to emit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compact_codec import ToolCall, parse_compact

TOOLS_PATH = Path(__file__).resolve().parent.parent / "tools.json"


@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    tool_calls: list[ToolCall]
    latency_ms: float


class FunctionGemmaModel:
    """Loads the fine-tuned FunctionGemma GGUF and emits compact tool calls."""

    def __init__(
        self,
        model_path: str,
        n_threads: int = 2,
        n_ctx: int = 2048,
        max_tokens: int = 64,
        cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        from llama_cpp import Llama, LlamaRAMCache  # lazy: wheel only on board

        self.model_path = model_path
        self.max_tokens = max_tokens
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        # Cache the tool-declaration prefix so turn 2+ skips ~1200 token
        # prefill on the 2-core A55 (drops cold ~48s -> warm ~1-3s).
        self.llm.set_cache(LlamaRAMCache(capacity_bytes=cache_bytes))
        self._tools: list[dict[str, Any]] = json.loads(TOOLS_PATH.read_text())["tools"]

    def generate(self, user_text: str) -> GenerationResult:
        prompt = self._build_prompt(user_text)
        t0 = time.time()
        # Use the low-level token loop because llama-cpp's create_completion
        # strips special tokens from the decoded output. The compact format
        # ``<tool_N>(args)<end>`` encodes the function name as a special
        # token, so we need detokenize(..., special=True) to see it.
        prompt_tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        new_tokens: list[int] = []
        for token_id in self.llm.generate(prompt_tokens, temp=0.0, top_p=1.0, top_k=1):
            if token_id == self.llm.token_eos():
                break
            new_tokens.append(token_id)
            if len(new_tokens) >= self.max_tokens:
                break
            decoded_so_far = self.llm.detokenize(new_tokens, special=True).decode(
                "utf-8", errors="replace",
            )
            # Only stop on the turn-level marker. <end> terminates a single
            # tool call; multi-tool sequences emit <tool_A>(...)<end><tool_B>(...)<end>,
            # so stopping at the first <end> would truncate legitimate output.
            if "<end_of_turn>" in decoded_so_far:
                break
        latency_ms = (time.time() - t0) * 1000.0
        raw = self.llm.detokenize(new_tokens, special=True).decode("utf-8", errors="replace")
        return GenerationResult(
            raw_text=raw,
            tool_calls=parse_compact(raw),
            latency_ms=latency_ms,
        )

    def _build_prompt(self, user_text: str) -> str:
        declarations = [self._tool_declaration(t) for t in self._tools]
        system = (
            "You are a model that can do function calling with the following functions\n"
            + "\n".join(declarations)
        )
        turns = [("developer", system), ("user", user_text)]
        prompt = "".join(
            f"<start_of_turn>{role}\n{content}\n<end_of_turn>\n" for role, content in turns
        )
        return prompt + "<start_of_turn>model\n"

    @staticmethod
    def _tool_declaration(tool: dict[str, Any]) -> str:
        fn = tool["function"]
        name = fn["name"]
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        props_str = ",".join(
            f"{k}:{{description:<escape>{v.get('description', '')}<escape>,"
            f"type:<escape>{v.get('type', 'STRING')}<escape>}}"
            for k, v in props.items()
        )
        req_str = ",".join(f"<escape>{r}<escape>" for r in required)

        return (
            "<start_function_declaration>\n"
            f"declaration:{name}{{"
            f"description:<escape>{desc}<escape>,"
            f"parameters:{{properties:{{{props_str}}},"
            f"required:[{req_str}],"
            f"type:<escape>OBJECT<escape>"
            f"}}}}\n"
            "<end_function_declaration>"
        )
