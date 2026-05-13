"""llama-cpp-python runtime for the fine-tuned FunctionGemma 270M model.

Octopus v2 inference. No tool schema is injected into the prompt — the
model routes from its learned functional-token weights. The prompt for each
turn is just::

    <start_of_turn>user
    {user_text}<end_of_turn>
    <start_of_turn>model

and the model emits::

    <tool_N>(arg1,arg2,...)<end>

which the compact codec parses back into ``ToolCall`` objects. The on-device
prompt is ~13 tokens, keeping cold prefill on the 2-core A55 around 0.5 s.

``tools.json`` is the source of truth for the dispatcher's arg validation
and is embedded as GGUF metadata for schema-drift checks. It is not loaded
into the inference prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from compact_codec import ToolCall, parse_compact

TOOLS_PATH = Path(__file__).resolve().parent / "tools.json"


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
        n_ctx: int = 1024,
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
        # Cache the user-turn prefix so turn 2+ skips re-tokenization. Tiny
        # win at v9's ~13-token prompts, but the call is cheap and harmless.
        self.llm.set_cache(LlamaRAMCache(capacity_bytes=cache_bytes))

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

    @staticmethod
    def _build_prompt(user_text: str) -> str:
        """v9 Octopus v2 prompt — user turn only, no developer/tools turn."""
        return (
            f"<start_of_turn>user\n"
            f"{user_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
