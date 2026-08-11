import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import notify_kv_cache_written
from vllm_ascend.ops.rotary_embedding import rope_forward_oot
from vllm_ascend.ops.shmem_runtime import (
    log_shmem_path_once,
    maybe_shmem_matmul_allreduce,
    maybe_shmem_matmul_reduce_scatter,
)
from vllm_ascend.ops.triton.muls_add import muls_add_triton
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.utils import enable_sp_by_pass, is_vl_model, npu_stream_switch, prefetch_stream


def _maybe_chunk_residual_impl(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    try:
        get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) != residual.size(0):
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            residual = F.pad(residual, (0, 0, 0, pad_size))
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]

    return residual


def _maybe_all_gather_and_maybe_unpad_impl(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x

    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)
    if flash_comm_v1_enabled and label:
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is None or not is_ep_comm:
            x = tensor_model_parallel_all_gather(x, 0)
            pad_size = _EXTRA_CTX.pad_size
            if pad_size > 0:
                x = x[:-pad_size]
        else:
            x = get_ep_group().all_gather(x, 0)
            if enable_sp_by_pass():  # TODO: do unpad
                return x
            # unpad
            num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
            result = torch.empty((num_tokens_across_dp_cpu.sum(), *x.shape[1:]), device=x.device, dtype=x.dtype)
            dp_size = get_dp_group().world_size
            x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
            offset = 0
            for idx in range(dp_size):
                num_tokens_dp = num_tokens_across_dp_cpu[idx]
                result[offset : offset + num_tokens_dp] = x[idx, :num_tokens_dp]
                offset += num_tokens_dp
            x = result

    return x


def _maybe_pad_and_reduce_impl(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    flash_comm_v1_enabled = getattr(forward_context, "flash_comm_v1_enabled", False) or (
        enable_sp_by_pass() and is_ep_comm
    )

    if not flash_comm_v1_enabled or (forward_context.is_draft_model and is_vl_model() and not is_ep_comm):
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        if enable_sp_by_pass():
            return get_ep_group().reduce_scatter(x.view(-1, *x.shape[1:]), 0)
        # padding
        dp_size = get_dp_group().world_size
        num_tokens_across_dp_cpu = get_forward_context().dp_metadata.num_tokens_across_dp_cpu
        padded_x = torch.empty((dp_size, _EXTRA_CTX.padded_length, *x.shape[1:]), device=x.device, dtype=x.dtype)
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            padded_x[idx, :num_tokens_dp] = x[offset : offset + num_tokens_dp]
            offset += num_tokens_dp

        return get_ep_group().reduce_scatter(padded_x.view(-1, *x.shape[1:]), 0)


def _maybe_all_gather_and_maybe_unpad_fake(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    if _EXTRA_CTX.flash_comm_v1_enabled and label:
        return torch.empty(
            (x.shape[0] * get_tensor_model_parallel_world_size(), *x.shape[1:]), device=x.device, dtype=x.dtype
        )

    return x


def _maybe_pad_and_reduce_fake(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    if _EXTRA_CTX.flash_comm_v1_enabled or enable_sp_by_pass():
        return torch.empty(
            (x.shape[0] // get_tensor_model_parallel_world_size(), *x.shape[1:]), device=x.device, dtype=x.dtype
        )

    return x


def _prefetch_preprocess_impl(weight: torch.Tensor, start_flag: torch.Tensor, max_weight_size: int) -> None:
    calculation_stream = torch_npu.npu.current_stream()
    weight_prefetch_stream = prefetch_stream()
    weight_prefetch_stream.wait_stream(calculation_stream)
    with npu_stream_switch(weight_prefetch_stream):
        maybe_npu_prefetch(inputs=weight, dependency=start_flag, max_size=max_weight_size)


def _prefetch_preprocess_impl_fake(weight: torch.Tensor, start_flag: torch.Tensor, max_weight_size: int) -> None:
    return


def _prefetch_postprocess_impl(stop_flag: torch.Tensor) -> None:
    calculation_stream = torch_npu.npu.current_stream()
    weight_prefetch_stream = prefetch_stream()
    calculation_stream.wait_stream(weight_prefetch_stream)


def _prefetch_postprocess_impl_fake(stop_flag: torch.Tensor) -> None:
    return


def _maybe_all_reduce_tensor_model_parallel_impl(final_hidden_states: torch.Tensor) -> torch.Tensor:
    moe_comm_type = _EXTRA_CTX.moe_comm_type
    if (
        moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
        or _EXTRA_CTX.flash_comm_v1_enabled
    ):
        return final_hidden_states
    else:
        return tensor_model_parallel_all_reduce(final_hidden_states)


def _matmul_and_reduce_impl(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.custom_op is not None
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    output = self.custom_op.matmul_and_reduce(input_parallel, bias_)

    return output


def _matmul_and_reduce_impl_fake(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    num_tokens = input_parallel.size(0)
    if _EXTRA_CTX.flash_comm_v1_enabled:
        num_tokens = num_tokens // self.tp_size
    output = torch.empty(
        size=(num_tokens, self.output_size_per_partition), device=input_parallel.device, dtype=input_parallel.dtype
    )

    return output


def _shmem_matmul_allreduce_impl(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    return maybe_shmem_matmul_allreduce(layer, input_parallel)


def _shmem_matmul_allreduce_impl_fake(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    output_shape = (*input_parallel.shape[:-1], layer.output_size_per_partition)
    return torch.empty(output_shape, device=input_parallel.device, dtype=input_parallel.dtype)


def _shmem_matmul_reduce_scatter_impl(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    return maybe_shmem_matmul_reduce_scatter(layer, input_parallel)


def _shmem_matmul_reduce_scatter_impl_fake(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    output_shape = (
        input_parallel.shape[0] // layer.tp_size,
        layer.output_size_per_partition,
    )
    return torch.empty(output_shape, device=input_parallel.device, dtype=input_parallel.dtype)


def _resolve_frontend_cache_layer(layer_name: str):
    forward_context = get_forward_context()
    no_compile_layers = getattr(forward_context, "no_compile_layers", None)
    if isinstance(no_compile_layers, dict) and layer_name in no_compile_layers:
        return no_compile_layers[layer_name]
    return get_current_vllm_config().compilation_config.static_forward_context.get(layer_name)


def _frontend_prefill_kv_cache_impl(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context = get_forward_context()
    per_layer_attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(per_layer_attn_metadata, dict):
        log_shmem_path_once(
            f"frontend-kv:{layer_name}:skip:no-per-layer-metadata",
            "frontend_kv_cache layer=%s action=skip reason=no_per_layer_metadata",
            layer_name,
        )
        return key, value

    attn_metadata = per_layer_attn_metadata.get(layer_name)
    if attn_metadata is None:
        log_shmem_path_once(
            f"frontend-kv:{layer_name}:skip:no-layer-metadata",
            "frontend_kv_cache layer=%s action=skip reason=no_layer_metadata",
            layer_name,
        )
        return key, value

    if attn_metadata.attn_state in (
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.PrefillNoCache,
    ):
        log_shmem_path_once(
            f"frontend-kv:{layer_name}:skip:state:{attn_metadata.attn_state.name}",
            "frontend_kv_cache layer=%s action=skip reason=state_%s",
            layer_name,
            attn_metadata.attn_state.name.lower(),
        )
        return key, value

    layer = _resolve_frontend_cache_layer(layer_name)
    if layer is None:
        log_shmem_path_once(
            f"frontend-kv:{layer_name}:skip:no-layer",
            "frontend_kv_cache layer=%s action=skip reason=layer_not_found",
            layer_name,
        )
        return key, value

    kv_cache = getattr(layer, "kv_cache", None)
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) <= 1:
        log_shmem_path_once(
            f"frontend-kv:{layer_name}:skip:no-kv-cache",
            "frontend_kv_cache layer=%s action=skip reason=kv_cache_unavailable",
            layer_name,
        )
        return key, value

    slot_mapping = getattr(attn_metadata, "slot_mapping", None)
    if slot_mapping is None:
        log_shmem_path_once(
            f"frontend-kv:{layer_name}:skip:no-slot-mapping",
            "frontend_kv_cache layer=%s action=skip reason=no_slot_mapping",
            layer_name,
        )
        return key, value

    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", key.shape[0]))
    layer.impl.do_kv_cache_update(
        layer,
        key[:num_actual_tokens],
        value[:num_actual_tokens],
        kv_cache,
        slot_mapping[:num_actual_tokens],
    )
    attn_metadata.kv_cache_written_by_frontend = True
    attn_metadata.reshape_cache_event = None
    notify_kv_cache_written()
    log_shmem_path_once(
        f"frontend-kv:{layer_name}:hit",
        "frontend_kv_cache layer=%s action=hit state=%s tokens=%d graph_capture=%s",
        layer_name,
        attn_metadata.attn_state.name.lower(),
        num_actual_tokens,
        bool(getattr(forward_context, "capturing", False)),
    )
    return key, value


def _frontend_prefill_kv_cache_impl_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return key, value


# TODO(Angazenn): The reason why we use a custom op to encapsulate npu_quantize
# is that aclnnAscendQuantV3(npu_quantize) use div_mode=False, while
# aclnnAddRmsNormQuantV2(npu_add_rms_norm_quant) use div_moe=True. We have to
# pass input_scale and input_scale_reciprocal at the same time to avoid redundant
# reciprocal calculation in fussion pass. We shall remove this once
# aclnnAddRmsNormQuantV2 supports div_moe=False.
def _quantize_impl(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


def _quantize_impl_fake(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


def _rope_forward_oot_impl_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


def _muls_add_impl_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="maybe_chunk_residual",
    op_func=_maybe_chunk_residual_impl,
    fake_impl=lambda x, residual: torch.empty_like(x),
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_all_gather_and_maybe_unpad",
    op_func=_maybe_all_gather_and_maybe_unpad_impl,
    fake_impl=_maybe_all_gather_and_maybe_unpad_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_pad_and_reduce",
    op_func=_maybe_pad_and_reduce_impl,
    fake_impl=_maybe_pad_and_reduce_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="prefetch_preprocess",
    op_func=_prefetch_preprocess_impl,
    fake_impl=_prefetch_preprocess_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="prefetch_postprocess",
    op_func=_prefetch_postprocess_impl,
    fake_impl=_prefetch_postprocess_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_all_reduce_tensor_model_parallel",
    op_func=_maybe_all_reduce_tensor_model_parallel_impl,
    fake_impl=lambda x: x,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="matmul_and_reduce",
    op_func=_matmul_and_reduce_impl,
    fake_impl=_matmul_and_reduce_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="shmem_matmul_allreduce",
    op_func=_shmem_matmul_allreduce_impl,
    fake_impl=_shmem_matmul_allreduce_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="shmem_matmul_reduce_scatter",
    op_func=_shmem_matmul_reduce_scatter_impl,
    fake_impl=_shmem_matmul_reduce_scatter_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="frontend_prefill_kv_cache",
    op_func=_frontend_prefill_kv_cache_impl,
    fake_impl=_frontend_prefill_kv_cache_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="quantize",
    op_func=_quantize_impl,
    fake_impl=_quantize_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="npu_rotary_embedding",
    op_func=rope_forward_oot,
    fake_impl=_rope_forward_oot_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="muls_add",
    op_func=muls_add_triton,
    fake_impl=_muls_add_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
