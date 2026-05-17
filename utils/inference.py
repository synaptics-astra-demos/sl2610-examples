# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Managed KV-cache inference runners for Torq runtime.

Provides cache-managing wrappers around VMFBInferenceRunner for 
self-attention (decoder-only) and encoder-decoder architectures, 
plus an ORT fallback runner.
"""

import logging
import os
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from time import perf_counter_ns

import numpy as np
import numpy.typing as npt
from torq.runtime import InferenceRunner, VMFBInferenceRunner
from torq.runtime.utils import TensorInfo
from iree.runtime import DeviceArray


class ORTInferenceRunner(InferenceRunner):
    """Inference runner backed by ONNX Runtime."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None,
    ):
        try:
            import onnxruntime as ort
        except ModuleNotFoundError:
            raise RuntimeError(
                "onnxruntime is not installed; install with `pip install onnxruntime`"
            )
        super().__init__(model_path)
        self._logger = logging.getLogger(self.__class__.__name__)

        self._opts = ort.SessionOptions()
        if isinstance(n_threads, int) and n_threads > 0:
            self._opts.intra_op_num_threads = n_threads
            self._opts.inter_op_num_threads = n_threads
        self._sess = ort.InferenceSession(
            self._model_path, self._opts, providers=["CPUExecutionProvider"]
        )
        self._inputs_info = [
            TensorInfo(None, i.shape) for i in self._sess.get_inputs()
        ]
        self._outputs_info = [
            TensorInfo(None, o.shape) for o in self._sess.get_outputs()
        ]
        self._logger.info("Loaded ONNX model '%s'", str(self._model_path))

    @property
    def inputs_info(self) -> list[TensorInfo] | None:
        return self._inputs_info

    @property
    def outputs_info(self) -> list[TensorInfo] | None:
        return self._outputs_info

    def _infer(
        self, inputs: list[np.ndarray] | dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        if isinstance(inputs, list):
            inputs = dict(
                zip([inp.name for inp in self._sess.get_inputs()], inputs)
            )
        return [np.asarray(o) for o in self._sess.run(None, inputs)]


class BaseManagedCacheRunner(VMFBInferenceRunner):
    """Abstract base for inference runners with managed KV caches."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        cache_start_idx: int = 1,
        **kwargs,
    ) -> None:
        kwargs["device_outputs"] = True
        super().__init__(model_path, **kwargs)

        if self.inputs_info is None or self.outputs_info is None:
            raise ValueError(
                f"Model '{model_path}' is missing input/output metadata "
                "required for KV cache management."
            )
        self._cache_start_idx = cache_start_idx

    @abstractmethod
    def reset_kv(self) -> None: ...

    @abstractmethod
    def save_kv_state(self): ...

    @abstractmethod
    def restore_kv_state(self, state) -> None: ...


class ManagedSelfAttnCacheRunner(BaseManagedCacheRunner):
    """VMFBInferenceRunner with managed self-attention KV cache.

    For decoder-only architectures (e.g. Gemma, LLaMA).
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        cache_start_idx: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(model_path, cache_start_idx=cache_start_idx, **kwargs)

        self._n_kv = len(self.outputs_info) - self._cache_start_idx
        in_info = self.inputs_info
        self._kv_init = [
            np.zeros(in_info[i].shape, dtype=np.dtype(in_info[i].dtype))
            for i in range(len(in_info) - self._n_kv, len(in_info))
        ]
        self._kv_cache = [self.allocate_device_array(z) for z in self._kv_init]

    def _infer(
        self, inputs: Iterable[npt.NDArray] | Mapping[str, npt.NDArray]
    ) -> list:
        if isinstance(inputs, Mapping):
            full_inputs = list(inputs.values()) + self._kv_cache
        else:
            full_inputs = list(inputs) + self._kv_cache

        results = super()._infer(full_inputs)
        for i in range(self._n_kv):
            self._kv_cache[i] = results[self._cache_start_idx + i]
        return results[: self._cache_start_idx]

    def reset_kv(self) -> None:
        self._kv_cache = [self.allocate_device_array(z) for z in self._kv_init]

    def save_kv_state(self) -> list[np.ndarray]:
        return [kv.to_host().copy() for kv in self._kv_cache]

    def restore_kv_state(self, state: list[np.ndarray]) -> None:
        self._kv_cache = [self.allocate_device_array(arr) for arr in state]

    def shift_kv(
        self,
        keep_last_n: int,
        seq_axis: int = 2,
        protect_first_n: int = 0,
    ) -> None:
        """Shift the last *keep_last_n* entries to just after the first
        *protect_first_n* positions, zeroing the rest."""
        for i in range(self._n_kv):
            host = self._kv_cache[i].to_host()
            seq_len = host.shape[seq_axis]
            dest_start = protect_first_n
            if dest_start + keep_last_n >= seq_len:
                continue
            new = np.zeros_like(host)
            if protect_first_n > 0:
                pfx = [slice(None)] * host.ndim
                pfx[seq_axis] = slice(0, protect_first_n)
                new[tuple(pfx)] = host[tuple(pfx)]
            src = [slice(None)] * host.ndim
            dst = [slice(None)] * host.ndim
            src[seq_axis] = slice(seq_len - keep_last_n, seq_len)
            dst[seq_axis] = slice(dest_start, dest_start + keep_last_n)
            new[tuple(dst)] = host[tuple(src)]
            self._kv_cache[i] = self.allocate_device_array(new)


class ManagedEncDecCacheRunner(BaseManagedCacheRunner):
    """VMFBInferenceRunner with managed self + cross attention KV cache.

    For encoder-decoder architectures (e.g. Moonshine).
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        initial_cache: list[npt.NDArray | DeviceArray] | None = None,
        cache_start_idx: int = 1,
        input_cache_start_idx: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model_path, cache_start_idx=cache_start_idx, **kwargs)

        self._input_cache_start: int = (
            input_cache_start_idx
            if input_cache_start_idx is not None
            else cache_start_idx
        )

        n_self_cache_outputs = len(self.outputs_info) - self._cache_start_idx
        if n_self_cache_outputs <= 0 or n_self_cache_outputs % 2 != 0:
            raise ValueError(
                f"Expected even number of self-attn cache outputs from index "
                f"{cache_start_idx}, got {n_self_cache_outputs}."
            )
        self._n_layers = n_self_cache_outputs // 2

        n_cache_inputs = len(self.inputs_info) - self._input_cache_start
        if n_cache_inputs != 4 * self._n_layers:
            raise ValueError(
                f"Expected {4 * self._n_layers} interleaved cache inputs, "
                f"got {n_cache_inputs}."
            )

        if initial_cache is not None:
            expected = 4 * self._n_layers
            if len(initial_cache) != expected:
                raise ValueError(
                    f"Expected {expected} cache tensors, got {len(initial_cache)}."
                )
            initial_cache = [
                self.allocate_device_array(c) if isinstance(c, np.ndarray) else c
                for c in initial_cache
            ]
        else:
            in_info = self.inputs_info
            initial_cache = [
                self.allocate_device_array(
                    np.zeros(
                        in_info[self._input_cache_start + i].shape,
                        dtype=np.dtype(in_info[self._input_cache_start + i].dtype),
                    )
                )
                for i in range(4 * self._n_layers)
            ]

        self._self_cache: list = [None] * (2 * self._n_layers)
        self._cross_cache: list = [None] * (2 * self._n_layers)
        for layer in range(self._n_layers):
            base = layer * 4
            self._self_cache[2 * layer] = initial_cache[base]
            self._self_cache[2 * layer + 1] = initial_cache[base + 1]
            self._cross_cache[2 * layer] = initial_cache[base + 2]
            self._cross_cache[2 * layer + 1] = initial_cache[base + 3]

    def set_cache(
        self,
        interleaved_cache: list[npt.NDArray | DeviceArray],
        *,
        pad_axis: int = 2,
    ) -> None:
        """Set self and cross caches from an interleaved list.

        Self-attention caches are zero-padded along *pad_axis* when their
        shape is smaller than the model expects.
        """
        expected = 4 * self._n_layers
        if len(interleaved_cache) != expected:
            raise ValueError(
                f"Expected {expected} cache tensors, got {len(interleaved_cache)}."
            )

        in_info = self.inputs_info
        for layer in range(self._n_layers):
            base = layer * 4
            for j in range(2):  # self v, k
                c = interleaved_cache[base + j]
                if isinstance(c, DeviceArray):
                    c = c.to_host()
                else:
                    c = np.asarray(c)
                target = tuple(
                    in_info[self._input_cache_start + base + j].shape
                )
                if c.shape != target:
                    padded = np.zeros(target, dtype=c.dtype)
                    slices = [slice(None)] * c.ndim
                    slices[pad_axis] = slice(0, c.shape[pad_axis])
                    padded[tuple(slices)] = c
                    c = padded
                self._self_cache[2 * layer + j] = self.allocate_device_array(c)

            for j in range(2):  # cross v, k
                c = interleaved_cache[base + 2 + j]
                if isinstance(c, DeviceArray):
                    c = c.to_host()
                else:
                    c = np.asarray(c)
                self._cross_cache[2 * layer + j] = self.allocate_device_array(c)

    def _infer(
        self, inputs: Iterable[npt.NDArray] | Mapping[str, npt.NDArray]
    ) -> list:
        interleaved = []
        for layer in range(self._n_layers):
            interleaved.append(self._self_cache[2 * layer])
            interleaved.append(self._self_cache[2 * layer + 1])
            interleaved.append(self._cross_cache[2 * layer])
            interleaved.append(self._cross_cache[2 * layer + 1])

        if isinstance(inputs, Mapping):
            full_inputs = list(inputs.values()) + interleaved
        else:
            full_inputs = list(inputs) + interleaved

        results = super()._infer(full_inputs)
        n_self_outputs = 2 * self._n_layers
        for i in range(n_self_outputs):
            self._self_cache[i] = results[self._cache_start_idx + i]
        return results[: self._cache_start_idx]

    def reset_kv(self) -> None:
        in_info = self.inputs_info
        for layer in range(self._n_layers):
            for j in range(2):
                idx = self._input_cache_start + layer * 4 + j
                info = in_info[idx]
                z = np.zeros(info.shape, dtype=np.dtype(info.dtype))
                self._self_cache[2 * layer + j] = self.allocate_device_array(z)

    def save_kv_state(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self_state = [c.to_host().copy() for c in self._self_cache]
        cross_state = [c.to_host().copy() for c in self._cross_cache]
        return self_state, cross_state

    def restore_kv_state(
        self, state: tuple[list[np.ndarray], list[np.ndarray]]
    ) -> None:
        self_state, cross_state = state
        self._self_cache = [self.allocate_device_array(a) for a in self_state]
        self._cross_cache = [self.allocate_device_array(a) for a in cross_state]


class SplitLMHeadRunner:
    """Adapter for split body + lm_head inference."""

    def __init__(
        self,
        body: BaseManagedCacheRunner,
        lm_head_path: str | os.PathLike,
        **kwargs,
    ) -> None:
        self._body = body
        self._infer_time_ms = 0.0
        lm_head_kwargs = {
            k: kwargs[k] for k in ("n_threads", "runtime_flags") if k in kwargs
        }
        self._lm_head = VMFBInferenceRunner(
            lm_head_path, device_outputs=True, **lm_head_kwargs
        )

    @property
    def model_path(self) -> str | os.PathLike:
        return self._body.model_path

    @property
    def infer_time_ms(self) -> float:
        return self._infer_time_ms

    @property
    def inputs_info(self):
        return self._body.inputs_info

    @property
    def outputs_info(self):
        body_outputs = self._body.outputs_info
        lm_head_outputs = self._lm_head.outputs_info
        if not body_outputs or not lm_head_outputs:
            return body_outputs
        return [lm_head_outputs[0], *body_outputs[1:]]

    @property
    def device(self):
        return self._body.device

    def infer(
        self, inputs: Iterable[npt.NDArray] | Mapping[str, npt.NDArray]
    ) -> list:
        start = perf_counter_ns()
        results = self._body.infer(inputs)
        lm_out = self._lm_head.infer([results[0]])
        self._infer_time_ms = (perf_counter_ns() - start) / 1e6
        return [lm_out[0], *results[1:]]

    def allocate_device_array(self, array: npt.NDArray) -> DeviceArray:
        return self._body.allocate_device_array(array)

    def reset_kv(self) -> None:
        self._body.reset_kv()

    def save_kv_state(self):
        return self._body.save_kv_state()

    def restore_kv_state(self, state) -> None:
        self._body.restore_kv_state(state)

    def shift_kv(self, *args, **kwargs) -> None:
        self._body.shift_kv(*args, **kwargs)
