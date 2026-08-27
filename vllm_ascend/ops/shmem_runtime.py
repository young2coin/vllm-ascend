import atexit
import importlib
import os
import threading
from typing import Any, Optional

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config

logger = init_logger(__name__)

_DEFAULT_BLOCK_DIMS = 20
_DEFAULT_LOCAL_MEM_SIZE = 1024 * 1024 * 1024
_DEFAULT_IP_PORT = "tcp://127.0.0.1:8667"
_OUTPUT_BUFFER_ALIGNMENT = 512
_MAX_SUPPORTED_RANKS = 8
_CACHED_BLOCK_DIMS: Optional[int] = None
_KERNEL_NAME_BY_DTYPE = {
    torch.bfloat16: "shmem_matmul_allreduce_overlap_bf16",
}


def _shmem_trace_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_SHMEM_TRACE", "").lower() in {
        "1",
        "on",
        "true",
        "yes",
    }


def shmem_matmul_allreduce_enabled() -> bool:
    enabled = bool(envs_ascend.VLLM_ASCEND_ENABLE_SHMEM_MATMUL_ALLREDUCE)
    if enabled and get_ascend_config().enable_matmul_allreduce:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_SHMEM_MATMUL_ALLREDUCE and "
            "the native enable_matmul_allreduce option cannot be enabled "
            "together"
        )
    return enabled


def _strip_tcp_prefix(ip_port: str) -> str:
    if ip_port.startswith("tcp://"):
        return ip_port[len("tcp://") :]
    return ip_port


def _get_block_dims() -> int:
    global _CACHED_BLOCK_DIMS
    if _CACHED_BLOCK_DIMS is None:
        _CACHED_BLOCK_DIMS = int(
            os.getenv("VLLM_ASCEND_SHMEM_BLOCK_DIMS", str(_DEFAULT_BLOCK_DIMS))
        )
    return _CACHED_BLOCK_DIMS


def _current_stream_handle() -> int:
    current_stream = torch.npu.current_stream()
    stream_handle = getattr(current_stream, "npu_stream", None)
    if stream_handle is None:
        raise RuntimeError("current stream does not expose npu_stream")
    return int(stream_handle)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * torch.empty((), dtype=dtype).element_size()


def _get_configured_output_buffer_bytes() -> int:
    value = os.getenv("VLLM_ASCEND_SHMEM_OUTPUT_BUFFER_BYTES")
    if value is None or value == "":
        return 0
    buffer_bytes = int(value)
    if buffer_bytes <= 0:
        raise RuntimeError(
            "VLLM_ASCEND_SHMEM_OUTPUT_BUFFER_BYTES must be positive"
        )
    return _align_up(buffer_bytes, _OUTPUT_BUFFER_ALIGNMENT)


def _get_prealloc_output_tokens() -> int:
    value = os.getenv("VLLM_ASCEND_SHMEM_OUTPUT_MAX_TOKENS")
    if value is not None and value != "":
        max_tokens = int(value)
        if max_tokens <= 0:
            raise RuntimeError(
                "VLLM_ASCEND_SHMEM_OUTPUT_MAX_TOKENS must be positive"
            )
        return max_tokens

    return int(get_current_vllm_config().scheduler_config.max_num_batched_tokens)


def _get_min_matmul_allreduce_tokens() -> int:
    value = os.getenv("VLLM_ASCEND_SHMEM_MATMUL_ALLREDUCE_MIN_TOKENS")
    if value is None or value == "":
        return 1
    min_tokens = int(value)
    if min_tokens < 1:
        raise RuntimeError(
            "VLLM_ASCEND_SHMEM_MATMUL_ALLREDUCE_MIN_TOKENS must be positive"
        )
    return min_tokens


def _is_graph_capturing() -> bool:
    if not is_forward_context_available():
        return False
    return bool(getattr(get_forward_context(), "capturing", False))


def _build_weight_for_shmem(layer: torch.nn.Module) -> torch.Tensor:
    cached = getattr(layer, "_shmem_matmul_allreduce_weight_t", None)
    weight_t_view = layer.weight.transpose(0, 1)
    if (
        cached is None
        or cached.shape != weight_t_view.shape
        or cached.dtype != weight_t_view.dtype
        or cached.device != weight_t_view.device
    ):
        cached = weight_t_view.contiguous()
        setattr(layer, "_shmem_matmul_allreduce_weight_t", cached)
    else:
        cached.copy_(weight_t_view)
    return cached


def prepare_shmem_matmul_allreduce(layer: torch.nn.Module) -> None:
    weight = getattr(layer, "weight", None)
    reason = None
    kernel_name = _KERNEL_NAME_BY_DTYPE.get(getattr(weight, "dtype", None))

    if weight is None:
        reason = "missing_weight"
    elif get_ascend_config().weight_nz_mode == 2:
        reason = "unsupported_nz_layout"
    elif weight.ndim != 2:
        reason = "weight_rank_ne_2"
    elif kernel_name is None:
        reason = f"unsupported_weight_dtype:{getattr(weight, 'dtype', None)}"

    setattr(layer, "_shmem_static_reason", reason)
    setattr(layer, "_shmem_kernel_name", kernel_name)
    setattr(layer, "_shmem_block_dims", _get_block_dims())
    min_tokens = _get_min_matmul_allreduce_tokens()
    setattr(layer, "_shmem_min_matmul_allreduce_tokens", min_tokens)
    setattr(layer, "_shmem_kernel_entry", None)
    setattr(layer, "_shmem_can_implement_entry", None)
    setattr(layer, "_shmem_matmul_allreduce_weight_t", None)


class _SymmetricOutputBuffer:
    def __init__(
        self,
        ash: Any,
        tensor_from_ptr: Any,
        dtype: torch.dtype,
        device: torch.device,
        requested_bytes: int,
    ) -> None:
        configured_bytes = _get_configured_output_buffer_bytes()
        self.buffer_bytes = max(
            _align_up(requested_bytes, _OUTPUT_BUFFER_ALIGNMENT),
            configured_bytes,
        )
        self.dtype = dtype
        self.device = device
        self._ash = ash
        self._tensor_from_ptr = tensor_from_ptr
        self._ptr = int(ash.aclshmem_malloc(self.buffer_bytes) or 0)
        self._tensors: dict[tuple[int, ...], torch.Tensor] = {}

        if not self._ptr:
            raise RuntimeError(
                "aclshmem_malloc failed for shmem output buffer: "
                f"buffer_bytes={self.buffer_bytes}"
            )

    def _ensure_tensor_for_shape(self, shape: tuple[int, ...]) -> None:
        if shape not in self._tensors:
            self._tensors[shape] = self._tensor_from_ptr(
                self._ptr,
                shape,
                self.dtype,
                self.device,
            )

    def make_tensor(self, shape: tuple[int, ...], requested_bytes: int) -> torch.Tensor:
        if requested_bytes > self.buffer_bytes:
            raise RuntimeError(
                "shmem output shape exceeds fixed buffer capacity: "
                f"requested_bytes={requested_bytes} "
                f"buffer_bytes={self.buffer_bytes}. "
                "Set VLLM_ASCEND_SHMEM_OUTPUT_BUFFER_BYTES to the maximum "
                "captured output size."
            )
        if shape not in self._tensors:
            if _is_graph_capturing():
                raise RuntimeError(
                    "shmem output tensor wrapper was not prepared before "
                    "graph capture: "
                    f"shape={shape}. Run a non-graph warmup for this "
                    "capture shape before ACL graph capture."
                )
            self._ensure_tensor_for_shape(shape)
        return self._tensors[shape]

    def free(self) -> None:
        self._tensors.clear()
        if self._ptr:
            self._ash.aclshmem_free(self._ptr)
            self._ptr = 0


class _ShmemRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._ash = None
        self._tensor_from_ptr = None
        self._shmem_operators = None
        self._operators: dict[int, object] = {}
        self._kernel_entries: dict[tuple[int, str], Any] = {}
        self._output_buffers: dict[tuple[torch.dtype, str], _SymmetricOutputBuffer] = {}
        self._printed_operator_call = False
        self._rank: Optional[int] = None
        self._world_size: Optional[int] = None

    def ensure_initialized(self, tp_rank: int, tp_size: int) -> Optional[str]:
        with self._lock:
            if self._initialized:
                if self._rank != tp_rank or self._world_size != tp_size:
                    return (
                        "shmem_runtime_group_changed:"
                        f"initialized={self._rank}/{self._world_size},"
                        f"requested={tp_rank}/{tp_size}"
                    )
                return None

            try:
                ash = importlib.import_module("shmem")
            except ImportError as exc:
                return f"missing_shmem_runtime:{exc}"

            try:
                shmem_operators = importlib.import_module("shmem_operators")
            except ImportError as exc:
                return f"missing_shmem_operators:{exc}"

            if not dist.is_initialized():
                return "torch_distributed_not_initialized"

            global_rank = dist.get_rank()
            global_world_size = dist.get_world_size()
            if tp_size < 2 or tp_size > _MAX_SUPPORTED_RANKS:
                return f"unsupported_tp_size:{tp_size}"
            if global_rank != tp_rank or global_world_size != tp_size:
                return (
                    "only_global_tp_group_is_supported:"
                    f"global={global_rank}/{global_world_size},"
                    f"tp={tp_rank}/{tp_size}"
                )

            ip_port = os.getenv("VLLM_ASCEND_SHMEM_IP_PORT", _DEFAULT_IP_PORT)
            os.environ.setdefault("SHMEM_UID_SESSION_ID", _strip_tcp_prefix(ip_port))

            attr = ash.InitAttr()
            attr.my_rank = tp_rank
            attr.n_ranks = tp_size
            attr.local_mem_size = int(
                os.getenv(
                    "VLLM_ASCEND_SHMEM_LOCAL_MEM_SIZE",
                    str(_DEFAULT_LOCAL_MEM_SIZE),
                )
            )
            attr.ip_port = ip_port

            ret = ash.aclshmem_init(attr)
            if ret != 0:
                return f"aclshmem_init_failed:{ret}"

            tensor_from_ptr = getattr(ash, "construct_tensor_from_ptr", None)
            if tensor_from_ptr is None:
                try:
                    tensor_module = importlib.import_module("shmem.construct_tensor")
                    tensor_from_ptr = tensor_module.construct_tensor_from_ptr
                except (AttributeError, ImportError) as exc:
                    ash.aclshmem_global_exit(0)
                    return f"missing_construct_tensor_from_ptr:{exc}"

            self._ash = ash
            self._tensor_from_ptr = tensor_from_ptr
            self._shmem_operators = shmem_operators
            self._rank = tp_rank
            self._world_size = tp_size
            self._initialized = True
            return None

    def get_kernel_entry(self, block_dims: int, kernel_name: str):
        with self._lock:
            key = (block_dims, kernel_name)
            kernel_entry = self._kernel_entries.get(key)
            if kernel_entry is None:
                operator = self._operators.get(block_dims)
                if operator is None:
                    assert self._shmem_operators is not None
                    operator = self._shmem_operators.ShmemOperators(block_dims)
                    self._operators[block_dims] = operator
                kernel_entry = getattr(operator, kernel_name, None)
                self._kernel_entries[key] = kernel_entry
            return kernel_entry

    def log_operator_call_once(
        self,
        layer_name: str,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
    ) -> None:
        with self._lock:
            if self._printed_operator_call or self._rank != 0:
                return
            self._printed_operator_call = True
            world_size = self._world_size

        logger.warning(
            "[shmem-operator] called=1 initialized=1 "
            "rank=0/%s layer=%s shape=(%s,%s,%s) dtype=%s",
            world_size,
            layer_name,
            m,
            n,
            k,
            dtype,
        )

    def get_symmetric_output(
        self,
        layer: torch.nn.Module,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        del layer
        return self._get_symmetric_output(shape, dtype, device)

    def prepare_symmetric_output(
        self,
        layer: torch.nn.Module,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        del layer
        self._prepare_symmetric_output(shape, dtype, device)

    def _prepare_symmetric_output(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._get_symmetric_output(shape, dtype, device)

    def _get_symmetric_output(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        with self._lock:
            assert self._ash is not None
            assert self._tensor_from_ptr is not None
            device_id = device.index
            if device_id is None:
                device_id = torch.npu.current_device()
            normalized_device = torch.device(f"npu:{device_id}")
            key = (dtype, str(normalized_device))
            requested_bytes = _tensor_nbytes(shape, dtype)
            buffer = self._output_buffers.get(key)
            if buffer is None:
                if _is_graph_capturing():
                    raise RuntimeError(
                        "shmem output buffer was not allocated before graph "
                        "capture. Run a non-graph warmup before ACL graph "
                        "capture, or set VLLM_ASCEND_SHMEM_OUTPUT_BUFFER_BYTES "
                        "and initialize the runtime before capture."
                    )
                buffer = _SymmetricOutputBuffer(
                    self._ash,
                    self._tensor_from_ptr,
                    dtype,
                    normalized_device,
                    requested_bytes,
                )
                self._output_buffers[key] = buffer
            return buffer.make_tensor(shape, requested_bytes)

    def destroy(self) -> None:
        with self._lock:
            if not self._initialized or self._ash is None:
                return
            try:
                for buffer in self._output_buffers.values():
                    try:
                        buffer.free()
                    except Exception:
                        logger.exception("Failed to free shmem output buffer")
                self._output_buffers.clear()
                self._kernel_entries.clear()
                self._operators.clear()
                self._ash.aclshmem_global_exit(0)
            except Exception:
                logger.exception("Failed to shutdown shmem runtime cleanly")
            finally:
                self._initialized = False
                self._ash = None
                self._tensor_from_ptr = None
                self._shmem_operators = None
                self._rank = None
                self._world_size = None


_RUNTIME = _ShmemRuntime()
atexit.register(_RUNTIME.destroy)


def finalize_shmem_matmul_allreduce(layer: torch.nn.Module) -> None:
    if not getattr(layer, "_can_try_shmem_matmul_allreduce", False):
        return

    static_reason = getattr(layer, "_shmem_static_reason", None)
    if static_reason is not None:
        raise RuntimeError(
            f"shmem matmul-allreduce cannot initialize {layer.prefix}: "
            f"{static_reason}"
        )

    weight_t = _build_weight_for_shmem(layer)
    if getattr(layer, "_shmem_kernel_entry", None) is not None:
        return
    init_reason = _RUNTIME.ensure_initialized(layer.tp_rank, layer.tp_size)
    if init_reason is not None:
        raise RuntimeError(
            f"shmem matmul-allreduce cannot initialize {layer.prefix}: "
            f"{init_reason}"
        )
    layer._shmem_kernel_entry = _RUNTIME.get_kernel_entry(
        layer._shmem_block_dims, layer._shmem_kernel_name
    )
    if layer._shmem_kernel_entry is None:
        raise RuntimeError(
            "shmem_operators does not expose required kernel: "
            f"{layer._shmem_kernel_name}"
        )
    layer._shmem_can_implement_entry = _RUNTIME.get_kernel_entry(
        layer._shmem_block_dims, "shmem_matmul_allreduce_can_implement_bf16"
    )


def can_use_shmem_matmul_allreduce(
    layer: torch.nn.Module,
    input_parallel: torch.Tensor,
) -> bool:
    if getattr(layer, "_shmem_static_reason", None) is not None:
        return False
    weight_t = getattr(layer, "_shmem_matmul_allreduce_weight_t", None)
    if weight_t is None:
        return False
    if input_parallel.dtype != weight_t.dtype:
        return False
    if input_parallel.device != weight_t.device:
        return False
    if input_parallel.shape[-1] != weight_t.shape[0]:
        return False
    m = int(input_parallel.numel() // input_parallel.shape[-1])
    n = int(weight_t.shape[1])
    k = int(input_parallel.shape[-1])
    if torch.compiler.is_compiling():
        return True
    can_implement = getattr(layer, "_shmem_can_implement_entry", None)
    if can_implement is None:
        return False
    return bool(can_implement(m, n, k))


def maybe_shmem_matmul_allreduce(
    layer: torch.nn.Module,
    input_parallel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    static_reason = getattr(layer, "_shmem_static_reason", None)
    if static_reason is not None:
        raise RuntimeError(f"shmem matmul-allreduce disabled: {static_reason}")
    if bias is not None:
        raise RuntimeError(
            "shmem matmul-allreduce overlap returns symmetric output directly "
            "and does not support fused bias"
        )

    weight_t = getattr(layer, "_shmem_matmul_allreduce_weight_t", None)
    if weight_t is None:
        raise RuntimeError("shmem matmul-allreduce weight is not finalized")
    if input_parallel.dtype != weight_t.dtype:
        raise RuntimeError(
            "shmem matmul-allreduce requires input and weight to use the "
            "same dtype: "
            f"input_dtype={input_parallel.dtype} weight_dtype={weight_t.dtype}"
        )
    if input_parallel.device != weight_t.device:
        raise RuntimeError(
            "shmem matmul-allreduce requires input and weight on the same "
            "device: "
            f"input_device={input_parallel.device} "
            f"weight_device={weight_t.device}"
        )
    if input_parallel.shape[-1] != weight_t.shape[0]:
        raise RuntimeError(
            "shmem matmul-allreduce input/weight shape mismatch: "
            f"input_k={input_parallel.shape[-1]} weight_k={weight_t.shape[0]}"
        )

    can_implement = getattr(layer, "_shmem_can_implement_entry", None)
    if can_implement is None:
        raise RuntimeError(
            "shmem matmul-allreduce capability entry is not initialized"
        )
    m = int(input_parallel.numel() // input_parallel.shape[-1])
    n = int(weight_t.shape[1])
    k = int(input_parallel.shape[-1])
    if not bool(can_implement(m, n, k)):
        raise RuntimeError(
            "shmem matmul-allreduce cannot implement shape: "
            f"m={m} n={n} k={k}"
        )

    kernel_entry = getattr(layer, "_shmem_kernel_entry", None)
    if kernel_entry is None:
        raise RuntimeError("shmem matmul-allreduce kernel entry is not initialized")

    if input_parallel.is_contiguous():
        input_2d = input_parallel.reshape(-1, input_parallel.shape[-1])
    else:
        input_2d = input_parallel.contiguous().reshape(-1, input_parallel.shape[-1])

    stream_handle = _current_stream_handle()
    output_2d = _RUNTIME.get_symmetric_output(
        layer,
        (input_2d.shape[0], weight_t.shape[1]),
        input_2d.dtype,
        input_2d.device,
    )
    kernel_entry(
        input_2d.data_ptr(),
        weight_t.data_ptr(),
        output_2d.data_ptr(),
        input_2d.shape[0],
        weight_t.shape[1],
        input_2d.shape[1],
        stream_handle,
    )
    _RUNTIME.log_operator_call_once(
        str(layer.prefix),
        int(input_2d.shape[0]),
        int(weight_t.shape[1]),
        int(input_2d.shape[1]),
        input_2d.dtype,
    )
    output = output_2d.clone()
    return output.reshape(*input_parallel.shape[:-1], weight_t.shape[1])
