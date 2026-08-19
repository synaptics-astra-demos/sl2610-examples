# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path

logger = logging.getLogger(__name__)


class GemmaBackend(ABC):
    """Abstract interface shared by all Gemma backends."""

    @abstractmethod
    def stream_response(self, query: str) -> Generator[str, None, None]:
        """Yield progressively longer answer strings as tokens arrive."""
        ...

    @property
    @abstractmethod
    def last_infer_time_ms(self) -> float:
        """Total wall-clock inference time of the last call, in ms."""
        ...

    @property
    @abstractmethod
    def time_to_first_token_ms(self) -> float:
        """Time from start to first generated token, in ms."""
        ...

    @property
    @abstractmethod
    def last_n_input_tokens(self) -> int:
        ...

    @property
    @abstractmethod
    def last_n_output_tokens(self) -> int:
        ...

    @property
    def last_n_prefill_tokens(self) -> int:
        return self.last_n_input_tokens

    @property
    def last_prefill_tps(self) -> float:
        ttft_s = self.time_to_first_token_ms / 1000
        return self.last_n_prefill_tokens / ttft_s if ttft_s > 0 else 0.0


class GemmaTorq(GemmaBackend):
    """Compatibility facade over torq-examples Gemma3Static."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        max_seq_len: int | None = None,
        max_prompt_tokens: int | None = None,
        n_threads: int | None = None,
        instruct_model: bool = True,
        cache_keep_n: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 64,
        runtime_flags: list[str] | None = None,
        device_io: bool = False,
        sys_prompt: str | None = None,
        lm_head_path: str | os.PathLike | None = None,
        disable_lm_head: bool = False,
    ):
        from app_utils.torq_examples.gemma3.src.runner import Gemma3Static

        self._runner = Gemma3Static(
            model_path,
            max_seq_len=max_seq_len,
            max_prompt_tokens=max_prompt_tokens,
            n_threads=n_threads,
            instruct_model=instruct_model,
            cache_keep_n=cache_keep_n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            runtime_flags=runtime_flags,
            device_io=device_io,
            sys_prompt=sys_prompt,
            lm_head_path=lm_head_path,
            disable_lm_head=disable_lm_head,
        )
        self._last_n_input_tokens = 0
        self._last_n_prefill_tokens = 0

    @property
    def max_seq_len(self) -> int:
        return self._runner.max_seq_len

    @property
    def last_infer_time_ms(self) -> float:
        return self._runner.last_infer_time

    @property
    def time_to_first_token_ms(self) -> float:
        return self._runner.time_to_first_token

    @property
    def last_n_input_tokens(self) -> int:
        return self._last_n_input_tokens

    @property
    def last_n_output_tokens(self) -> int:
        return int(getattr(self._runner, "generated_tokens"))

    @property
    def last_n_prefill_tokens(self) -> int:
        return self._last_n_prefill_tokens

    def stream_response(self, query: str) -> Generator[str, None, None]:
        self._record_prompt_stats(query)
        full_text = ""
        for chunk in self._runner.run_stream(query):
            full_text += str(chunk)
            yield full_text

    def _record_prompt_stats(self, query: str) -> None:
        try:
            prompt_tokens = self._runner._build_prompt_tokens(query)
            prefill_tokens = self._runner._apply_prompt_limit(prompt_tokens)
        except Exception:
            prompt_tokens = self._fallback_prompt_tokens(query)
            prefill_tokens = prompt_tokens
        self._last_n_input_tokens = len(prompt_tokens)
        self._last_n_prefill_tokens = len(prefill_tokens)

    def _fallback_prompt_tokens(self, query: str) -> list[int]:
        if self._runner.is_instruct_model:
            return self._runner.tokenize(query, "user") + self._runner.tokenize(
                "", "model"
            )
        return self._runner.tokenize(query)


class GemmaLlama(GemmaBackend):
    """Gemma inference via llama-cpp-python (GGUF models)."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_ctx: int = 800,
        n_threads: int = 2,
        temperature: float = 0.2,
        max_tokens: int = 100,
    ):
        from llama_cpp import Llama

        self._logger = logging.getLogger(self.__class__.__name__)
        self._model_path = Path(model_path)
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            chat_format="gemma",
            verbose=False,
        )
        self._last_infer_ms: float = 0.0
        self._ttft_ms: float = 0.0
        self._n_input: int = 0
        self._n_output: int = 0

        self._logger.info("Loaded Gemma llama model '%s'", str(model_path))

    @property
    def last_infer_time_ms(self) -> float:
        return self._last_infer_ms

    @property
    def time_to_first_token_ms(self) -> float:
        return self._ttft_ms

    @property
    def last_n_input_tokens(self) -> int:
        return self._n_input

    @property
    def last_n_output_tokens(self) -> int:
        return self._n_output

    def stream_response(self, query: str) -> Generator[str, None, None]:
        self._n_input = len(self._llm.tokenize(query.encode()))
        answer_parts: list[str] = []
        self._n_output = 0
        first_token_time = None

        t_start = time.time()
        for chunk in self._llm.create_chat_completion(
            messages=[{"role": "user", "content": query}],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content")
            if token:
                if first_token_time is None:
                    first_token_time = time.time()
                    self._ttft_ms = (first_token_time - t_start) * 1000
                self._n_output += 1
                answer_parts.append(token)
                yield "".join(answer_parts)

        t_end = time.time()
        self._last_infer_ms = (t_end - t_start) * 1000

        final = "".join(answer_parts).strip()
        yield final


def load_gemma(
    *,
    use_llama: bool = False,
    model_path: str | os.PathLike | None = None,
    n_threads: int | None = None,
    **kwargs,
) -> GemmaBackend:
    """Instantiate a Gemma backend.

    Args:
        use_llama: If ``True``, use the llama.cpp backend; otherwise use
            the torq VMFB backend (default).
        model_path: Path to the model file/directory. For torq this is
            the ``.vmfb`` file; for llama this is the ``.gguf`` file.
            When ``None``, the torq backend uses the managed
            ``transformer.vmfb`` path (sibling ``lm_head.vmfb`` is
            auto-discovered).
        n_threads: Thread count for inference.
        **kwargs: Forwarded to the chosen backend constructor.

    Returns:
        A ``GemmaBackend`` instance.
    """
    if use_llama:
        if model_path is None:
            raise ValueError(
                "model_path is required for the llama.cpp backend "
                "(path to .gguf file)"
            )
        llama_kw = {k: kwargs[k] for k in ("n_ctx", "temperature", "max_tokens") if k in kwargs}
        return GemmaLlama(model_path, n_threads=n_threads or 2, **llama_kw)

    # Torq backend
    if model_path is None:
        from app_utils.paths import MODELS_DIR
        from app_utils.torq_examples.gemma3.setup_demo import (
            GEMMA3_HF_REPO_MAP,
            download_gemma3,
        )
        from app_utils.torq_examples.utils.download import resolve_repo_id

        repo_id = resolve_repo_id("instruct", GEMMA3_HF_REPO_MAP)
        model_dir = MODELS_DIR / repo_id
        model_path = model_dir / "transformer.vmfb"
        if not model_path.exists():
            download_gemma3(["instruct"], base_dir=MODELS_DIR)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Default Gemma model not found at '{model_path}'. "
                "Pass --gemma-model to use a different VMFB."
            )

    torq_kw = {
        k: kwargs[k]
        for k in (
            "max_seq_len", "max_prompt_tokens", "instruct_model",
            "cache_keep_n", "temperature", "top_p", "top_k",
            "runtime_flags", "device_io", "sys_prompt", "lm_head_path",
            "disable_lm_head",
        )
        if k in kwargs
    }
    return GemmaTorq(model_path, n_threads=n_threads, **torq_kw)
