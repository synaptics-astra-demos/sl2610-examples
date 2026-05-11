# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Moonshine speech-to-text inference runner.

Provides a clean, single-class API for running the Moonshine 
encoder-decoder pipeline on Torq VMFB models.
"""

import logging
import os
from pathlib import Path
from time import perf_counter_ns
from typing import Final, Union

import numpy as np

from torq.runtime import VMFBInferenceRunner

from ..inference import ORTInferenceRunner, ManagedEncDecCacheRunner

_START_TOKEN_ID: Final[int] = 1
_END_TOKEN_ID: Final[int] = 2
_DEFAULT_INPUT_FREQ: Final[int] = 16000

InferenceBackend = Union[ORTInferenceRunner, VMFBInferenceRunner]


def _find_models(model_dir: Path) -> dict[str, Path]:
    """Discover Moonshine model files in a directory."""
    known = {"preprocessor", "encoder", "decoder", "decoder_with_past"}
    models: dict[str, Path] = {}
    for f in model_dir.iterdir():
        if f.is_file() and f.suffix in (".vmfb", ".onnx") and f.stem in known:
            models[f.stem] = f
    return models


def _get_runner(
    model_path: Path,
    n_threads: int | None = None,
    runtime_flags: list[str] | None = None,
    *rargs,
    **rkwargs,
) -> InferenceBackend:
    model_path = Path(model_path)
    ext = model_path.suffix.lower()
    if ext == ".onnx":
        return ORTInferenceRunner(model_path, n_threads=n_threads)
    elif ext == ".vmfb":
        return VMFBInferenceRunner(
            model_path,
            n_threads=n_threads,
            runtime_flags=runtime_flags,
            *rargs,
            **rkwargs,
        )
    raise TypeError(f"Unsupported model format '{ext}'")


class MoonshineRunner:
    """Moonshine speech-to-text inference runner.

    Requires ``encoder.vmfb``, ``decoder.vmfb``, and
    ``decoder_with_past.vmfb`` in the model directory.
    An optional ``preprocessor.vmfb`` is applied before encoding,
    and ``decoder_token_embeddings.npy`` enables embedding-lookup
    input to the decoder.
    """

    __slots__ = (
        "_logger",
        "_preprocessor",
        "_encoder",
        "_decoder",
        "_decoder_cached",
        "_token_embeddings",
        "_max_inp_len",
        "_input_freq",
        "_n_tokens_gen",
        "_last_infer_ns",
        "_time_to_first_token_ns",
    )

    def __init__(
        self,
        model_dir: str | os.PathLike,
        *,
        input_freq: int = _DEFAULT_INPUT_FREQ,
        n_threads: int | None = None,
        runtime_flags: list[str] | None = None,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        model_dir = Path(model_dir)

        components = _find_models(model_dir)
        for req in ("encoder", "decoder", "decoder_with_past"):
            if req not in components:
                raise ValueError(
                    f"Missing required model '{req}.vmfb' in {model_dir}"
                )

        rk: dict = dict(n_threads=n_threads, runtime_flags=runtime_flags)

        self._preprocessor: InferenceBackend | None = (
            _get_runner(components["preprocessor"], **rk)
            if "preprocessor" in components
            else None
        )
        self._encoder = _get_runner(components["encoder"], **rk)
        self._decoder = _get_runner(components["decoder"], **rk)
        self._decoder_cached = ManagedEncDecCacheRunner(
            components["decoder_with_past"],
            input_cache_start_idx=2,  # [token_emb, seq_len, *cache]
            cache_start_idx=1,  # [logits, *self_cache]
            **rk,
        )

        self._token_embeddings = self._load_embeddings(model_dir)

        enc_info = (
            self._preprocessor.inputs_info
            if self._preprocessor
            else self._encoder.inputs_info
        )
        if enc_info:
            self._max_inp_len: int | None = int(enc_info[0].shape[-1])
        else:
            self._max_inp_len = None
            self._logger.warning(
                "Cannot determine max input length from encoder metadata; "
                "audio will not be padded/truncated."
            )

        self.input_freq = input_freq
        self._n_tokens_gen: int = 0
        self._last_infer_ns: int = 0
        self._time_to_first_token_ns: int = 0

        self._logger.info("Loaded Moonshine models from '%s'", model_dir)

    @property
    def last_infer_time(self) -> float:
        """Total inference time of the last ``run()`` call, in milliseconds."""
        return self._last_infer_ns / 1e6

    @property
    def time_to_first_token(self) -> float:
        """Encoder + first decoder step time, in milliseconds."""
        return self._time_to_first_token_ns / 1e6

    @property
    def generated_tokens(self) -> int:
        return self._n_tokens_gen

    @property
    def max_inp_len(self) -> int | None:
        return self._max_inp_len

    @property
    def input_freq(self) -> int:
        return self._input_freq

    @input_freq.setter
    def input_freq(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"input_freq must be positive, got {value}")
        self._input_freq = int(value)

    @staticmethod
    def _load_embeddings(model_dir: Path) -> np.ndarray | None:
        paths = list(model_dir.glob("*token_embeddings.npy"))
        if not paths:
            return None
        arr = np.load(paths[0])
        if arr.dtype == np.dtype("V2"):
            import ml_dtypes

            arr = arr.view(ml_dtypes.bfloat16)
        return arr

    def _size_input(self, audio: np.ndarray) -> np.ndarray:
        """Pad or truncate audio to ``max_inp_len``."""
        if self._max_inp_len is None:
            return audio
        audio = audio.flatten()
        if len(audio) > self._max_inp_len:
            self._logger.debug(
                "Truncating input from %d to %d", len(audio), self._max_inp_len
            )
            audio = audio[: self._max_inp_len]
        elif len(audio) < self._max_inp_len:
            self._logger.debug(
                "Padding input from %d to %d", len(audio), self._max_inp_len
            )
            audio = np.pad(audio, (0, self._max_inp_len - len(audio)))
        return audio.reshape(1, -1)

    def _get_token_input(self, token_id: int) -> np.ndarray:
        """Embedding lookup or raw token ID for decoder input."""
        if self._token_embeddings is not None:
            return np.expand_dims(self._token_embeddings[token_id], axis=(0, 1))
        return np.array([[token_id]], dtype=np.int32)

    def _run_encoder(self, audio: np.ndarray) -> np.ndarray:
        if self._preprocessor is not None:
            audio = self._preprocessor.infer([audio])[0]
        enc_info = self._encoder.inputs_info
        if enc_info:
            audio = audio.astype(np.dtype(enc_info[0].dtype))
        enc_out = self._encoder.infer([audio])[0]
        self._logger.debug(
            "Encoder infer: %.3f ms", self._encoder.infer_time_ms
        )
        return enc_out

    def run(
        self,
        audio: np.ndarray,
        *,
        max_tokens: int | None = None,
    ) -> np.ndarray:
        """Transcribe audio to a token array.

        Args:
            audio: Raw audio waveform, shape ``(1, N)`` or ``(N,)``.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            Token IDs as ``np.ndarray`` with shape ``(1, T)``.
        """
        self._n_tokens_gen = 0
        audio = self._size_input(audio)

        if max_tokens is None:
            max_tokens = max(1, int((audio.shape[-1] / self._input_freq) * 6))

        start_ns = perf_counter_ns()

        encoder_out = self._run_encoder(audio)
        dec_info = self._decoder.inputs_info
        if dec_info and len(dec_info) > 1:
            encoder_out = encoder_out.astype(np.dtype(dec_info[1].dtype))

        # first decoder step -> initial cache
        token_emb = self._get_token_input(_START_TOKEN_ID)
        if dec_info:
            token_emb = token_emb.astype(np.dtype(dec_info[0].dtype))

        results = self._decoder.infer([token_emb, encoder_out])
        self._logger.debug(
            "First decoder infer: %.3f ms", self._decoder.infer_time_ms
        )
        logits = results[0]
        initial_cache = results[1:]
        self._decoder_cached.set_cache(initial_cache)

        next_token = int(np.asarray(logits)[0, -1].argmax())
        self._n_tokens_gen += 1
        self._time_to_first_token_ns = perf_counter_ns() - start_ns
        tokens = [_START_TOKEN_ID, next_token]

        # autoregressive decoding with cached decoder
        dec_cached_info = self._decoder_cached.inputs_info
        for i in range(max_tokens - 1):
            if next_token == _END_TOKEN_ID:
                break
            token_emb = self._get_token_input(next_token)
            seq_len = np.array([[i + 1]], dtype=np.int32)
            if dec_cached_info:
                token_emb = token_emb.astype(np.dtype(dec_cached_info[0].dtype))
                seq_len = seq_len.astype(np.dtype(dec_cached_info[1].dtype))
            [logits] = self._decoder_cached.infer([token_emb, seq_len])
            self._logger.debug(
                "Cached decoder infer: %.3f ms",
                self._decoder_cached.infer_time_ms,
            )
            next_token = int(np.asarray(logits)[0, -1].argmax())
            self._n_tokens_gen += 1
            tokens.append(next_token)

        self._last_infer_ns = perf_counter_ns() - start_ns
        return np.array([tokens])
