import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from math import floor
from pathlib import Path
from typing import Final, Literal

import numpy as np
import onnxruntime as ort
import ml_dtypes

from .filesystem import (
    StaticMemoryKVCache,
    StaticFileKVCache,
    UniqueTempDir
)
from .runners import (
    run_vmfb,
    InferenceRunner,
    ORTInferenceRunner,
    IREEInferenceRunner,
    IREE_RT_AVAILABLE
)


class MoonshineStaticBase(ABC):

    def __init__(
        self,
        model_size: Literal["base", "tiny"],
        max_inp_len: int,
        max_dec_len: int
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._model_size = model_size
        self._max_inp_len = max_inp_len
        self._max_dec_len = max_dec_len
        self._enc_out_len = floor(floor(floor(self._max_inp_len / 64 - 127 / 64) / 3) / 2) - 1

        if self._model_size == "tiny":
            self._n_layers: int = 6
            self._n_kv_heads: int = 8
            self._head_dim: int = 36
        else:
            self._n_layers: int = 8
            self._n_kv_heads: int = 8
            self._head_dim: int = 52
        self._start_token_id: int = 1
        self._end_token_id: int = 2
        self._encoder_pad_id: int = 2

        self._n_tokens_gen: int = 0
        self._infer_times: deque[int] = deque(maxlen=100)

    @property
    def last_infer_time(self) -> float:
        return self._infer_times[-1] if self._infer_times else 0.0

    @property
    def avg_infer_time(self) -> float:
        return (sum(self._infer_times) / len(self._infer_times)) if self._infer_times else 0.0

    @property
    def max_inp_len(self) -> int:
        return self._max_inp_len

    def _size_input(self, input: np.ndarray) -> np.ndarray:
        input = input.flatten()
        if len(input) > self._max_inp_len:
            self._logger.debug("Truncating input from %d to %d", len(input), self.max_inp_len)
            input = input[: self._max_inp_len]
        elif len(input) < self._max_inp_len:
            self._logger.debug("Padding input from %d to %d", len(input), self.max_inp_len)
            input = np.pad(
                input,
                (0, self._max_inp_len - len(input)),
                constant_values=self._encoder_pad_id,
            )
        return input.reshape((1, self._max_inp_len))

    @abstractmethod
    def _run_encoder(self, input: np.ndarray) -> Path | np.ndarray: ...

    @abstractmethod
    def _run_decoder(
        self,
        input_tokens: list[int],
        encoder_out: Path | np.ndarray, *, seq_len: int
    ) -> int: ...

    def run(
        self,
        input: np.ndarray,
        max_tokens: int | None = None,
    ) -> np.ndarray:
        import time

        self._n_tokens_gen = 0
        max_tokens = (
            max_tokens
            if isinstance(max_tokens, int) and max_tokens < self._max_dec_len
            else self._max_dec_len
        )

        st = time.time()
        next_token = self._start_token_id
        tokens = [next_token]
        input = self._size_input(input)
        encoder_out = self._run_encoder(input)

        next_token = self._run_decoder(tokens, encoder_out, seq_len=0)
        self._n_tokens_gen += 1
        tokens.append(next_token)

        for i in range(max_tokens):
            next_token = self._run_decoder(
                [next_token], encoder_out, seq_len=i + 1
            )
            self._n_tokens_gen += 1
            tokens.append(next_token)
            if next_token == self._end_token_id:
                break

        self._infer_times.append(time.time() - st)
        return np.array([tokens])


class MoonshineStatic(MoonshineStaticBase):

    def __init__(
        self,
        encoder: InferenceRunner,
        decoder: InferenceRunner,
        decoder_with_past: InferenceRunner,
        model_size: Literal["base", "tiny"],
        max_inp_len: int,
        max_dec_len: int,
        preprocessor: InferenceRunner | None = None,
    ):
        super().__init__(
            model_size,
            max_inp_len,
            max_dec_len
        )

        self._encoder = encoder
        self._decoder = decoder
        self._decoder_with_past = decoder_with_past
        self._preprocessor = preprocessor
        self._kv_cache = StaticMemoryKVCache(
            self._n_layers,
            self._n_kv_heads,
            self._enc_out_len,
            self._max_dec_len,
            self._head_dim
        )
        self._token_embeddings: np.ndarray | None = self._find_token_embeddings()
        self._dec_int_inp_type = np.int64 if isinstance(self._decoder_with_past, ORTInferenceRunner) else np.int32

    @classmethod
    def from_onnx(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        decoder_with_past_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        n_threads: int | None = None
    ) -> "MoonshineStatic":
        encoder_ort = ort.InferenceSession(encoder_model, providers=['CPUExecutionProvider'])
        max_inp_len: int = next(
            inp.shape for inp in encoder_ort.get_inputs() if inp.name == "input_values"
        )[-1]
        decoder_with_past_ort = ort.InferenceSession(decoder_with_past_model, providers=['CPUExecutionProvider'])
        max_dec_len: int = next(
            inp.shape
            for inp in decoder_with_past_ort.get_inputs()
            if "decoder" in inp.name
        )[2]  # assuming shape [B, H, L, D]

        return cls(
            ORTInferenceRunner(encoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_model, n_threads=n_threads),
            ORTInferenceRunner(decoder_with_past_model, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len
        )

    @classmethod
    def from_vmfb(
        cls,
        encoder_model: str | os.PathLike,
        decoder_model: str | os.PathLike,
        decoder_with_past_model: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        max_inp_len: int,
        max_dec_len: int,
        n_threads: int | None = None,
        device: str = "torq",
    ) -> "MoonshineStatic":
        return cls(
            IREEInferenceRunner(encoder_model, device_uri=device, n_threads=n_threads),
            IREEInferenceRunner(decoder_model, device_uri=device, n_threads=n_threads),
            IREEInferenceRunner(decoder_with_past_model, device_uri=device, n_threads=n_threads),
            model_size,
            max_inp_len,
            max_dec_len
        )

    def _find_token_embeddings(
        self,
        emb_pattern: str = "decoder*token_embeddings.npy",
    ) -> np.ndarray | None:
        paths = []
        for model in (self._decoder, self._decoder_with_past):
            paths.extend(model.model_path.parent.glob(emb_pattern))
        if not paths:
            return None

        paths = list({p.resolve(): p for p in paths}.values())
        if len(paths) > 1:
            raise RuntimeError(
                f"Expected a single token embedding file, found {len(paths)}: {paths}"
            )

        arr = np.load(paths[0])
        if arr.dtype == np.dtype('V2'):
            arr = arr.view(ml_dtypes.bfloat16)
        return arr

    def _run_preprocessor(
        self, input: np.ndarray,
    ):
        if self._preprocessor is None:
            return input
        return self._preprocessor.infer([input])[0]

    def _run_encoder(
        self, input: np.ndarray
    ) -> np.ndarray:
        input = self._run_preprocessor(input)
        encoder_out = self._encoder.infer([input])[0]
        if isinstance(self._decoder, IREEInferenceRunner):
            encoder_out = encoder_out.astype(np.dtype(ml_dtypes.bfloat16))
        return encoder_out

    def _get_decoder_token_input(
        self,
        input_tokens: list[int]
    ) -> np.ndarray:
        if isinstance(self._token_embeddings, np.ndarray):
            last_tok = input_tokens[-1]
            return np.expand_dims(self._token_embeddings[last_tok], axis=(0, 1))
        return np.array([input_tokens], dtype=self._dec_int_inp_type)

    def _run_decoder(
        self, input_tokens: list[int], encoder_out: np.ndarray, *, seq_len: int
    ) -> int:
        token_input = self._get_decoder_token_input(input_tokens)
        if seq_len == 0:
            logits, *cache = self._decoder.infer([
                token_input,
                encoder_out
            ])
            self._kv_cache.update_cache(cache, update_all=True)
            self._kv_cache.pad_self_cache() # because decoder outputs BxHx1xD caches
        else:
            logits, *cache = self._decoder_with_past.infer([
                token_input,
                *self._kv_cache.all_cache,
                np.array([[seq_len]], dtype=self._dec_int_inp_type),
            ])
            self._kv_cache.update_cache(cache)

        next_token = logits[0, -1].argmax().item()
        return next_token


class MoonshineStaticIreeCli(MoonshineStaticBase):

    def __init__(
        self,
        encoder: str | os.PathLike,
        decoder: str | os.PathLike,
        decoder_with_past: str | os.PathLike,
        model_size: Literal["base", "tiny"],
        max_inp_len: int,
        max_dec_len: int,
        *,
        n_threads: int | None = None
    ):
        super().__init__(
            model_size,
            max_inp_len,
            max_dec_len
        )

        self._encoder = encoder
        self._decoder = decoder
        self._decoder_with_past = decoder_with_past
        self._n_threads = n_threads
        self._kv_cache = StaticFileKVCache(
            self._n_layers,
            self._n_kv_heads,
            self._enc_out_len,
            self._max_dec_len,
            self._head_dim
        )
        self._work_dir = UniqueTempDir().dir_path

    def _run_encoder(
        self, input: np.ndarray
    ) -> Path:
        inp_input = self._work_dir / "input.npy"
        np.save(inp_input, input)
        out_encoder = self._work_dir / "last_hidden_states.npy"
        run_vmfb(
            self._encoder,
            [f"@{str(inp_input)}"],
            [f"@{str(out_encoder)}"],
            n_threads=self._n_threads
        )
        return out_encoder

    def _run_decoder(
        self, input_tokens: list[int], encoder_out: Path, *, seq_len: int
    ) -> int:
        inp_input_ids = self._work_dir / "input_ids.npy"
        out_logits = self._work_dir / "logits.npy"
        np.save(inp_input_ids, np.array([input_tokens], dtype=np.int64))

        if seq_len == 0:
            inputs = [
                f"@{str(f)}"
                for f in (inp_input_ids, encoder_out)
            ]
            outputs = [
                f"@{str(out_logits)}",
                *[f"@{str(cache)}" for cache in self._kv_cache.all_cache]
            ]
            run_vmfb(
                self._decoder,
                inputs,
                outputs,
                n_threads=self._n_threads
            )
            self._kv_cache.pad_self_cache() # because decoder outputs BxHx1xD caches
        else:
            inp_current_len = self._work_dir / "current_len.npy"
            np.save(inp_current_len, np.array([[seq_len]], dtype=np.int64))
            inputs = [
                f"@{str(inp_input_ids)}",
                *[f"@{str(cache)}" for cache in self._kv_cache.all_cache],
                f"@{str(inp_current_len)}"
            ]
            outputs = [
                f"@{str(out_logits)}",
                *[f"@{str(cache)}" for cache in self._kv_cache.self_cache]
            ]
            run_vmfb(
                self._decoder_with_past,
                inputs,
                outputs,
                n_threads=self._n_threads
            )

        next_token = np.load(out_logits)[0, -1].argmax().item()
        return next_token


def find_moonshine_models(
    model_dir: str | os.PathLike,
) -> dict[str, Path | None]:
    model_dir = Path(model_dir)
    components = {
        "preprocessor": None,
        "encoder": None,
        "decoder": None,
        "decoder_with_past": None
    }
    for f in model_dir.iterdir():
        if not f.is_file() or f.suffix not in (".onnx", ".vmfb"):
            continue
        if f.stem in components.keys():
            components[f.stem] = f
    return components


def load_moonshine(
    model_dir: str | os.PathLike,
    model_size: str,
    max_inp_len: int | None,
    max_dec_len: int | None,
    use_cli: bool = False,
    n_threads: int | None = None,
    device: str = "torq",
) -> MoonshineStatic:

    def _get_runner(model_path: Path, *rargs, **rkwargs):
        model_path = Path(model_path)
        model_type = model_path.suffix.lower()
        if model_type == ".onnx":
            return ORTInferenceRunner(
                model_path, *rargs, **rkwargs, n_threads=n_threads
            )
        elif model_type == ".vmfb":
            return IREEInferenceRunner(
                model_path, *rargs, **rkwargs, n_threads=n_threads, device_uri=device
            )

    components: dict[str, Path | None] = find_moonshine_models(model_dir)
    if any(components[c] is None for c in ("encoder", "decoder", "decoder_with_past")):
        raise ValueError("Missing one or more moonshine models from ['encoder', 'decoder', 'decoder_with_past']")
    
    has_vmfb = any(Path(components[c]).suffix.lower() == ".vmfb" for c in components if components[c] is not None)
    has_onnx = any(Path(components[c]).suffix.lower() == ".onnx" for c in components if components[c] is not None)
    all_onnx = all(Path(components[c]).suffix.lower() == ".onnx" for c in ("encoder", "decoder", "decoder_with_past"))
    
    if has_vmfb and (not max_inp_len or not max_dec_len):
        raise ValueError(
            f"Valid maximum input length and maximum decoder length are required for static VMFB models, received ({max_inp_len}, {max_dec_len})"
        )
    
    # Use MoonshineStatic if:
    # 1. All models are ONNX (ORT doesn't need IREE), OR
    # 2. IREE runtime is available (can handle both ONNX and VMFB)
    if (all_onnx or IREE_RT_AVAILABLE) and not use_cli:
        return MoonshineStatic(
            _get_runner(components["encoder"]),
            _get_runner(components["decoder"]),
            _get_runner(components["decoder_with_past"]),
            model_size,
            max_inp_len,
            max_dec_len,
            preprocessor=components["preprocessor"],
        )
    # CLI runner only supports VMFB files
    if has_onnx:
        if has_vmfb:
            raise ValueError(
                "Mixed ONNX/VMFB models detected but IREE runtime is not available. "
                "Either use all ONNX models or install iree-runtime for VMFB support."
            )
        raise ValueError(
            "iree-run-module runner is only available for full VMFB runtime. "
            "ONNX models require IREE runtime to be installed."
        )
    return MoonshineStaticIreeCli(
        _get_runner(components["encoder"]),
        _get_runner(components["decoder"]),
        _get_runner(components["decoder_with_past"]),
        model_size,
        max_inp_len,
        max_dec_len,
        n_threads=n_threads
    )


def format_answer(
    answer: str,
    infer_time: float,
    stats: list[str] | None = None,
    agent_name: str = "Agent"
) -> str:
    GREEN: Final[str] = "\033[32m"
    RESET: Final[str] = "\033[0m"
    result: str = GREEN + f"{agent_name}: {answer}" + RESET + f" ({infer_time * 1000:.3f} ms"
    stats = stats or []
    for stat in stats:
        result += ", " + str(stat)
    return result + ")"
