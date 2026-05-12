import logging
import os
import time
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
try:
    import iree.runtime as iree_rt
    print("Got IREE module", str(iree_rt.__file__))
    IREE_RT_AVAILABLE = True
except ImportError:
    print("No IREE module")
    IREE_RT_AVAILABLE = False

from .download import download_from_hf


def run_vmfb(
    model_path: str | os.PathLike,
    inputs: list[str],
    outputs: list[str],
    device: str = "torq",
    n_threads: int | None = None,
) -> float:
    n_threads = n_threads if (isinstance(n_threads, int) and n_threads > 0) else os.cpu_count()
    cmd: list[str] = [
        "./iree-run-module",
        f"--device={device}",
        f"--task_topology_group_count={n_threads}",
        f"--module={model_path}"
    ]
    cmd.extend([f"--input={inp}" for inp in inputs])
    cmd.extend([f"--output={out}" for out in outputs])

    st = time.perf_counter_ns()
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.DEVNULL,   # discard all stdout
        stderr=subprocess.PIPE       # capture stderr only
    )
    et = time.perf_counter_ns()

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to run VMFB model '{str(model_path)}':\n"
            + result.stderr.strip()
        )

    return (et - st) / 1e6


class InferenceRunner(ABC):

    def __init__(
        self,
        model_path: str | os.PathLike,
    ):
        self._model_path: Path = Path(model_path)
        self._infer_time_ms: float = 0.0

    @property
    def model_path(self) -> str | os.PathLike:
        return self._model_path

    @property
    def infer_time_ms(self) -> float:
        return self._infer_time_ms

    @abstractmethod
    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        ...

    def infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        st = time.perf_counter_ns()
        results = self._infer(inputs)
        self._infer_time_ms = (time.perf_counter_ns() - st) / 1e6
        return results


class ORTInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None
    ):
        super().__init__(model_path)
        self._logger = logging.getLogger(__class__.__name__)

        self._opts = ort.SessionOptions()
        if isinstance(n_threads, int) and n_threads > 0:
            self._opts.intra_op_num_threads = n_threads
            self._opts.inter_op_num_threads = n_threads
            self._logger.debug("Using %d threads for inference", n_threads)
        self._sess = ort.InferenceSession(self._model_path, self._opts, providers=['CPUExecutionProvider'])
        self._logger.info("Loaded ONNX model '%s'", str(self._model_path))

    @classmethod
    def from_hf(cls, hf_repo: str, filename: str | os.PathLike, **kwargs):
        model_path = download_from_hf(hf_repo, filename)
        return cls(model_path, **kwargs)

    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        if isinstance(inputs, list):
            inputs = dict(zip([inp.name for inp in self._sess.get_inputs()], inputs))
        st = time.perf_counter_ns()
        result = [np.asarray(o) for o in self._sess.run(None, inputs)]
        self._logger.debug("Infer '%s': %.3f ms", str(self._model_path), (time.perf_counter_ns() - st) / 1e6)
        return result


class IREEInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None,
        function: str = "main",
        device_uri: str = "torq",
        load_method: Literal["preload", "mmap"] = "preload",
        load_model_to_mem: bool = True
    ):
        if not IREE_RT_AVAILABLE:
            raise RuntimeError("IREE Runtime python API not available in environment")

        super().__init__(model_path)
        self._logger = logging.getLogger(__class__.__name__)
        self._function = function
        self._device_uri = device_uri
        self._load_method = load_method

        flags = set()
        if isinstance(n_threads, int) and n_threads > 0:
            flags.add(f"--task_topology_group_count={n_threads}")
            flags.add(f"--task_topology_max_group_count={n_threads}")
            self._logger.debug("Using %d threads for inference", n_threads)
        if device_uri == "torq":
            function = "main"
            flags.add("--torq_hw_type=astra_machina")
        
        if flags:
            iree_rt.flags.parse_flags(*flags)

        self._invoker = None
        if load_model_to_mem:
            self._invoker = self._load_invoker()

    def _load_invoker(self):
        if self._load_method == "mmap":
            module = iree_rt.load_vm_flatbuffer_file(self._model_path, driver=self._device_uri)
            vm_module = module.vm_module
            self._logger.debug("'%s' loaded via mmap", str(self._model_path))
        else:
            instance = iree_rt.VmInstance()
            device = iree_rt.get_device(self._device_uri)
            hal_module = iree_rt.create_hal_module(instance, device)
            with open(self._model_path, "rb") as f:
                fb = f.read()
            vm_module = iree_rt.VmModule.from_flatbuffer(instance, fb, warn_if_copy=False)
            _ctx = iree_rt.SystemContext(vm_modules=[hal_module, vm_module])
            module = _ctx.modules[vm_module.name]
            # Keep strong Python references so close() can destroy them in order.
            # Without these, the Python wrappers are destroyed when _load_invoker returns,
            # leaving orphaned C++ objects held only by nanobind keep_alive records on
            # self._invoker — which causes "nanobind: leaked N instances!" on exit.
            self._iree_instance = instance
            self._iree_device = device
            self._iree_hal_module = hal_module
            self._iree_vm_module = vm_module
            self._iree_ctx = _ctx
            self._logger.debug("'%s' loaded via memory copy", str(self._model_path))

        if self._function not in vm_module.function_names:
            raise ValueError(f"Function '{self._function}' not found in '{self._model_path}'")
        self._logger.info("Loaded VMFB model '%s'", str(self._model_path))
        return module[self._function]

    def close(self):
        """Explicitly release IREE runtime objects to prevent nanobind leak warnings on exit."""
        self._invoker = None
        for attr in ('_iree_ctx', '_iree_vm_module', '_iree_hal_module', '_iree_device', '_iree_instance'):
            if hasattr(self, attr):
                setattr(self, attr, None)

    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        if isinstance(inputs, dict):
            inputs = list(inputs.values())
        if self._invoker is None:
            self._invoker = self._load_invoker()
        st = time.perf_counter_ns()
        result: iree_rt.DeviceArray | tuple[iree_rt.DeviceArray] = self._invoker(*inputs)
        self._logger.debug("Infer '%s': %.3f ms", str(self._model_path), (time.perf_counter_ns() - st) / 1e6)
        if isinstance(result, tuple):
            return [r.to_host() for r in result]
        return [result.to_host()]
