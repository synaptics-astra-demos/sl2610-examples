import atexit
import os
import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from math import prod
from pathlib import Path
from typing import Any, MutableSequence

import numpy as np


class StaticKVCache(ABC):

    def __init__(
        self,
        layers:   int,
        kv_heads: int,
        enc_len:  int,
        dec_len:  int,
        head_dim: int,
        batch_sz: int = 1    
    ):
        self._layers   = layers
        self._kv_heads = kv_heads
        self._enc_len  = enc_len
        self._dec_len  = dec_len
        self._head_dim = head_dim
        self._batch_sz = batch_sz

        self._self_attn_shape: tuple[int, int, int, int]  = (
            self._batch_sz, self._kv_heads, self._dec_len, self._head_dim
        )
        self._cross_attn_shape: tuple[int, int, int, int] = (
            self._batch_sz, self._kv_heads, self._enc_len, self._head_dim
        )
        self._slot_shapes: tuple[tuple[int, int, int, int]] = (
            self._self_attn_shape,   # self.key
            self._self_attn_shape,   # self.value
            self._cross_attn_shape,  # cross.key
            self._cross_attn_shape,  # cross.value
        )
        self._self_cache, self._all_cache = self._init_cache()

    @property
    def all_cache(self) -> MutableSequence[Any]:
        return self._all_cache

    @property
    def self_cache(self) -> MutableSequence[Any]:
        return self._self_cache

    @abstractmethod
    def _init_cache(self) -> tuple[MutableSequence[Any], MutableSequence[Any]]: ...

    @abstractmethod
    def pad_self_cache(self): ...


class StaticMemoryKVCache(StaticKVCache):

    def __init__(
        self,
        layers:   int,
        kv_heads: int,
        enc_len:  int,
        dec_len:  int,
        head_dim: int,
        batch_sz: int = 1
    ):
        super().__init__(
            layers, kv_heads, enc_len, dec_len, head_dim, batch_sz
        )
    
    def _init_cache(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self_cache: list[np.ndarray] = []
        all_cache: list[np.ndarray] = []

        for _ in range(self._layers):
            for slot_idx, shape in enumerate(self._slot_shapes):
                p = np.zeros(shape, dtype=np.float32)
                if slot_idx < 2:
                    self_cache.append(p)
                all_cache.append(p)

        return self_cache, all_cache

    def _update_all_from_self(self):
        layer_idx = 0
        for self_idx, cache in enumerate(self._self_cache):
            all_idx = self_idx + 2 * layer_idx
            self._all_cache[all_idx] = cache
            if self_idx % 2:
                layer_idx += 1

    def pad_self_cache(self):
        for self_idx, cache in enumerate(self._self_cache):
            padded = np.pad(
                cache,
                pad_width=((0, 0), (0, 0), (0, self._self_attn_shape[2] - cache.shape[2]), (0, 0)),
                mode="constant"
            )
            all_idx = 2 * self_idx - (self_idx % 2)
            self._self_cache[self_idx] = padded
            self._all_cache[all_idx]   = padded

    def update_cache(self, new_values: list[np.ndarray], update_all: bool = False):
        to_update = self._all_cache if update_all else self._self_cache
        if len(to_update) != len(new_values):
            raise ValueError(f"Cache mismatch: expected {len(to_update)} new values, received {len(new_values)}")
        
        if update_all:
            for all_idx, new_value in enumerate(new_values):
                self_idx = None if all_idx % 4 > 1 else all_idx // 2 + all_idx % 2
                if self_idx is not None:    # exclude cross-attn caches
                    self._self_cache[self_idx] = new_value
                self._all_cache[all_idx] = new_value
        else:
            for self_idx, new_value in enumerate(new_values):
                all_idx = 2 * self_idx - (self_idx % 2)
                self._self_cache[self_idx] = new_value
                self._all_cache[all_idx]   = new_value


class StaticFileKVCache(StaticKVCache):

    def __init__(
        self,
        layers:   int,
        kv_heads: int,
        enc_len:  int,
        dec_len:  int,
        head_dim: int,
        batch_sz: int = 1
    ):
        super().__init__(
            layers, kv_heads, enc_len, dec_len, head_dim, batch_sz
        )

    def _init_cache(self) -> tuple[list[Path], list[Path]]:

        itemsize = np.dtype(np.float32).itemsize
        cache_sz_bytes = (
            2 * self._layers * prod(self._self_attn_shape)  * itemsize +   # self.k, self.v
            2 * self._layers * prod(self._cross_attn_shape) * itemsize     # cross.k, cross.v
        )

        cache_dir = UniqueTempDir(required_bytes=cache_sz_bytes).dir_path
        self_cache_paths: list[Path] = []
        all_cache_paths: list[Path] = []

        for layer_idx in range(self._layers):
            for slot_idx, shape in enumerate(self._slot_shapes):
                global_idx = layer_idx * 4 + slot_idx
                p = cache_dir / f"cache-{global_idx}.npy"
                np.save(p, np.zeros(shape, dtype=np.float32))
                if slot_idx < 2:
                    self_cache_paths.append(p)
                all_cache_paths.append(p)

        return self_cache_paths, all_cache_paths

    def pad_self_cache(self):
        for cache_path in self.self_cache:
            cache = np.load(cache_path)
            cache = np.pad(
                cache,
                pad_width=((0, 0), (0, 0), (0, self._self_attn_shape[2] - cache.shape[2]), (0, 0)),
                mode="constant"
            )
            np.save(cache_path, cache)


class UniqueTempDir:

    _registry: set[Path] = set()
    _lock: threading.Lock = threading.Lock()
    _atexit_registered: bool = False

    def __init__(
        self,
        shm_path: str = "/dev/shm",
        prefix: str = "tmp-",
        required_bytes: int | None = None,
        safety_factor: float = 1.2
    ):
        self._shm_path = shm_path
        self._prefix = prefix
        self._required_bytes = required_bytes
        self._safety_factor = safety_factor
        self._path: Path | None = None
        if not type(self)._atexit_registered:
            atexit.register(type(self).cleanup)
            type(self)._atexit_registered = True

    @classmethod
    def cleanup(cls):
        with cls._lock:
            for p in list(cls._registry):
                shutil.rmtree(p, ignore_errors=True)
            cls._registry.clear()

    @property
    def dir_path(self) -> Path:
        if self._path is None:
            base = Path(tempfile.gettempdir())
            try:
                if os.name == "posix" and os.path.isdir(self._shm_path) and os.access(self._shm_path, os.W_OK):
                    if self._required_bytes is not None:
                        _, _, free = shutil.disk_usage(self._shm_path)
                        need = int(self._required_bytes * self._safety_factor)
                        if free >= need:
                            base = Path(self._shm_path)
                    base = Path(self._shm_path)
            except Exception:
                pass
            p = Path(tempfile.mkdtemp(dir=str(base), prefix=self._prefix))
            with type(self)._lock:
                type(self)._registry.add(p)
            self._path = p
        return self._path

    def _cleanup_temp_dir(self):
        p = self._path
        if p is None:
            return
        with type(self)._lock:
            type(self)._registry.discard(p)
        shutil.rmtree(p, ignore_errors=True)
        self._path = None

    def __call__(self) -> Path:
        return self.dir_path

    def __enter__(self) -> Path:
        return self.dir_path

    def __exit__(self, exc_type, exc, tb):
        self._cleanup_temp_dir()
