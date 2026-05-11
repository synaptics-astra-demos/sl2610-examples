# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Gemma LLM backends with a common streaming interface.

Provides two interchangeable backends:
- ``GemmaTorq``: VMFB model via torq.runtime (default, ported from
  torq-examples/gemma3/src/runner.py).
- ``GemmaLlama``: GGUF model via llama-cpp-python (legacy fallback).

Use ``load_gemma()`` to instantiate the appropriate backend.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import Final

import ml_dtypes
import numpy as np
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

DEFAULT_SYS_PROMPT: Final[str] = (
    "You are a helpful AI assistant named Gemma. "
    "Answer in 1-2 sentences. No lists, no bullet points, no repetition."
)


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


class GemmaTorq(GemmaBackend):
    """Gemma inference via torq.runtime VMFB models.

    Ported from torq-examples ``Gemma3Static``.
    """

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
        sys_prompt: str | None = None,
        lm_head_path: str | os.PathLike | None = None,
    ):
        from ..inference import ManagedSelfAttnCacheRunner, SplitLMHeadRunner

        self._logger = logging.getLogger(self.__class__.__name__)

        self._model = ManagedSelfAttnCacheRunner(
            model_path,
            n_threads=n_threads,
            runtime_flags=runtime_flags,
        )
        if lm_head_path is not None:
            self._model = SplitLMHeadRunner(
                self._model,
                lm_head_path,
                n_threads=n_threads,
                runtime_flags=runtime_flags,
            )

        model_seq_len = self._query_model_seq_len()
        if max_seq_len is not None and model_seq_len is not None:
            if max_seq_len != model_seq_len:
                self._logger.warning(
                    "max_seq_len=%d vs model KV dim=%d; using %d",
                    max_seq_len, model_seq_len, model_seq_len,
                )
            max_seq_len = model_seq_len
        elif max_seq_len is None and model_seq_len is not None:
            max_seq_len = model_seq_len
        elif max_seq_len is None:
            raise ValueError(
                "Cannot determine max_seq_len: pass it explicitly."
            )

        self._model_dir = Path(self._model.model_path).parent
        with open(self._model_dir / "config.json") as f:
            cfg = json.load(f)
        self._n_layers: int = cfg["num_hidden_layers"]
        self._n_kv_heads: int = cfg["num_key_value_heads"]
        self._head_dim: int = cfg["head_dim"]
        self._bos_token_id: int = cfg["bos_token_id"]
        self._eos_token_id: int = cfg["eos_token_id"]
        self._pad_token_id: int = cfg.get("pad_token_id") or 0

        self._tokenizer = Tokenizer.from_file(
            str(self._model_dir / "tokenizer.json")
        )
        self._nl_token_id: int = self._tokenizer.encode("\n").ids[-1]
        self._double_nl_token_id: int = self._tokenizer.encode("\n\n").ids[-1]
        self._end_of_turn_id: int = self._tokenizer.token_to_id(
            "<end_of_turn>"
        )

        self._max_prompt_tokens = max_prompt_tokens
        self._max_seq_len = max_seq_len
        self._max_user_tokens: int | None = None
        self._instruct_model = instruct_model
        if instruct_model:
            self._sys_prompt = sys_prompt or DEFAULT_SYS_PROMPT
        else:
            self._sys_prompt = None
        self._cache_keep_n = cache_keep_n
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k

        self._token_embeddings = self._load_embeddings()
        self._token_id_lut = self._load_token_id_lut()
        if self._token_id_lut is not None:
            self._logger.info(
                "Loaded token ID LUT (%d entries)", len(self._token_id_lut)
            )
        self._pos_buf = np.zeros((1, 1), dtype=np.int32)
        if self._token_embeddings is not None:
            self._emb_buf = np.zeros(
                (1, 1, self._token_embeddings.shape[-1]),
                dtype=self._token_embeddings.dtype,
            )
        else:
            self._emb_buf = None

        # Warmup with system prompt
        self._warmup_len = self._warmup() if instruct_model else 0
        if self._warmup_len > 0:
            self._reset_cache_state = self._model.save_kv_state()
        else:
            self._reset_cache_state = []

        # Stats
        self._n_input_tokens: int = 0
        self._n_tokens_gen: int = 0
        self._last_infer_ns: int = 0
        self._time_to_first_token_ns: int = 0

        self._logger.info("Loaded Gemma torq model '%s'", str(model_path))

    @property
    def last_infer_time_ms(self) -> float:
        return self._last_infer_ns / 1e6

    @property
    def time_to_first_token_ms(self) -> float:
        return self._time_to_first_token_ns / 1e6

    @property
    def last_n_input_tokens(self) -> int:
        return self._n_input_tokens

    @property
    def last_n_output_tokens(self) -> int:
        return self._n_tokens_gen

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def _load_embeddings(self) -> np.ndarray | None:
        paths = list(self._model_dir.glob("token_embeddings.npy"))
        if not paths:
            return None
        arr = np.load(paths[0], mmap_mode="r")
        if arr.dtype == np.dtype("V2"):
            arr = arr.view(ml_dtypes.bfloat16)
        return arr

    def _load_token_id_lut(self) -> np.ndarray | None:
        paths = list(self._model_dir.glob("token_id_lut.npy"))
        if not paths:
            return None
        return np.load(paths[0])

    def _query_model_seq_len(self) -> int | None:
        info = self._model.inputs_info
        if info is None or len(info) < 3:
            return None
        kv_shape = info[2].shape
        if len(kv_shape) >= 3 and isinstance(kv_shape[2], int):
            return kv_shape[2]
        return None

    def _reset_cache(self):
        if self._reset_cache_state:
            self._model.restore_kv_state(self._reset_cache_state)
        else:
            self._model.reset_kv()

    def _tokenize(self, text: str, role: str | None = None) -> list[int]:
        if not self._instruct_model or role is None:
            return self._tokenizer.encode(text).ids
        # Gemma 3 chat format: <start_of_turn>role\ntext<end_of_turn>\n
        # BOS is added once at warmup start; strip auto-prepended BOS here.
        if role == "model":
            ids = self._tokenizer.encode("<start_of_turn>model\n").ids
        else:
            ids = self._tokenizer.encode(
                f"<start_of_turn>{role}\n{text}<end_of_turn>\n"
            ).ids
        if ids and ids[0] == self._bos_token_id:
            ids = ids[1:]
        return ids

    def _llm_step(self, token: int, seq_pos: int, *, sample: bool = True) -> int:
        if self._emb_buf is not None:
            self._emb_buf[0, 0, :] = self._token_embeddings[token]
            first = self._emb_buf
        else:
            self._pos_buf[0, 0] = token
            first = self._pos_buf.copy()

        self._pos_buf[0, 0] = seq_pos
        results = self._model.infer([first, self._pos_buf])

        if not sample:
            return 0
        compact_idx = self._sample(results[0].to_host()[0, -1])
        if self._token_id_lut is not None:
            return int(self._token_id_lut[compact_idx])
        return compact_idx

    def _sample(self, logits: np.ndarray) -> int:
        logits = logits.astype(np.float32, copy=False)
        if self._temperature <= 0:
            return int(logits.argmax())

        k = min(self._top_k, logits.shape[-1])
        top_k_idx = np.argpartition(logits, -k)[-k:]
        x = logits[top_k_idx]
        x /= self._temperature
        x -= x.max()
        np.exp(x, out=x)
        x /= x.sum()

        order = np.argsort(x)[::-1]
        cdf = np.cumsum(x[order])
        cut = int(np.searchsorted(cdf, self._top_p)) + 1
        keep = order[:cut]
        p = x[keep]
        p /= p.sum()
        return int(np.random.choice(top_k_idx[keep], p=p))

    def _prefill(self, tokens: list[int], start: int = 0) -> tuple[int, int]:
        pos = start
        for tok_id in tokens[:-1]:
            self._llm_step(tok_id, pos, sample=False)
            pos += 1
        if tokens:
            tok = self._llm_step(tokens[-1], pos)
        else:
            tok = 0
        pos += 1
        return tok, pos

    def _stop(self, token: int, gen: list[int]) -> bool:
        if token == self._eos_token_id:
            return True
        if self._end_of_turn_id is not None and token == self._end_of_turn_id:
            return True
        if not self._instruct_model and len(gen) > 2:
            if token == self._double_nl_token_id:
                return True
            return all(t == self._nl_token_id for t in gen[-2:])
        return False

    def _warmup(self) -> int:
        self._logger.info("Warm-up started...")
        sys_tokens = [self._bos_token_id] + self._tokenize(
            self._sys_prompt, "system"
        )
        if isinstance(self._max_prompt_tokens, int):
            sys_tokens = sys_tokens[: self._max_prompt_tokens]
            self._max_user_tokens = max(
                0, self._max_prompt_tokens - len(sys_tokens)
            )
        n = len(sys_tokens)
        self._prefill(sys_tokens)
        self._logger.info(
            "Warm-up complete: %d system tokens, %d remaining capacity",
            n, self._max_seq_len - n,
        )
        return n

    def stream_response(self, query: str) -> Generator[str, None, None]:
        self._reset_cache()
        self._n_tokens_gen = 0
        self._last_infer_ns = 0
        self._time_to_first_token_ns = 0

        tokens = self._tokenize(query, "user")
        if self._instruct_model:
            tokens += self._tokenize("", "model")

        self._n_input_tokens = len(tokens)

        limit = (
            self._max_user_tokens
            if self._max_user_tokens is not None
            else self._max_prompt_tokens
        )
        if isinstance(limit, int):
            if len(tokens) > limit:
                tokens = tokens[:limit]
            elif len(tokens) < limit:
                tokens += [self._pad_token_id] * (limit - len(tokens))

        gen: list[int] = []
        start_ns = time.perf_counter_ns()
        yield_ns = 0  # time spent suspended at yield (consumer time)
        try:
            next_tok, pos = self._prefill(tokens, start=self._warmup_len)
            self._time_to_first_token_ns = time.perf_counter_ns() - start_ns

            gen = [next_tok]
            full_text = self._tokenizer.decode(gen)
            _t = time.perf_counter_ns()
            yield full_text
            yield_ns += time.perf_counter_ns() - _t

            while not self._stop(next_tok, gen):
                if pos >= self._max_seq_len:
                    if self._cache_keep_n is not None:
                        self._model.shift_kv(
                            self._cache_keep_n,
                            protect_first_n=self._warmup_len,
                        )
                        pos = self._warmup_len + self._cache_keep_n
                    else:
                        self._logger.warning("Max generation tokens reached")
                        break
                next_tok = self._llm_step(next_tok, pos)
                gen.append(next_tok)
                pos += 1
                full_text = self._tokenizer.decode(gen)
                _t = time.perf_counter_ns()
                yield full_text
                yield_ns += time.perf_counter_ns() - _t
        finally:
            self._n_tokens_gen = max(0, len(gen) - 1)
            self._last_infer_ns = (time.perf_counter_ns() - start_ns) - yield_ns


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
            When ``None``, the torq backend downloads from HuggingFace.
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
        from .download import download_gemma3

        dirs = download_gemma3(["instruct"])
        model_dir = dirs["instruct"]
        # Prefer model.vmfb, fall back to model.vmfb.trim
        if (model_dir / "model.vmfb.trim").exists():
            model_path = model_dir / "model.vmfb.trim"
        elif (model_dir / "model.vmfb").exists():
            model_path = model_dir / "model.vmfb"
        else:
            raise FileNotFoundError(
                f"No model.vmfb or model.vmfb.trim found in {model_dir}"
            )

    torq_kw = {
        k: kwargs[k]
        for k in (
            "max_seq_len", "max_prompt_tokens", "instruct_model",
            "cache_keep_n", "temperature", "top_p", "top_k",
            "runtime_flags", "sys_prompt", "lm_head_path",
        )
        if k in kwargs
    }
    return GemmaTorq(model_path, n_threads=n_threads, **torq_kw)
