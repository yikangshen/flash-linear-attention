# -*- coding: utf-8 -*-

import os
from typing import List

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_product import chunk_gated_delta_product
from fla.ops.gated_delta_product.chunk_ref import chunk_gated_delta_product_ref
from fla.ops.gated_delta_product.naive import naive_recurrent_gated_delta_product
from fla.utils import assert_close, device, is_intel_alchemist


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'num_householder', 'gate_logit_normalizer', 'mask_p', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scale{}-num_householder{}-gate_logit_normalizer{}-mask_p{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, 0.1, 1, 1, 0, torch.float16),
            (2, 200, 3, 60, 0.1, 1, 1, 0, torch.float16),
            (2, 1000, 4, 64, 0.1, 2, 0.1, 0.5, torch.float16),
            (2, 1024, 4, 64, 1, 2, 1, 0, torch.float16),
            (2, 1024, 6, 100, 1, 2, 10, 0, torch.float16),
            (4, 1500, 8, 128, 0.1, 3, 1, 0.5, torch.float16),
            (2, 2048, 8, 128, 1, 3, 1, 0, torch.float16),
            (2, 2048, 8, 128, 1, 3, 1, 0, torch.float16),
        ]
    ]
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    num_householder: int,
    gate_logit_normalizer: float,
    mask_p: float,
    dtype: torch.dtype,
):
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')

    q = torch.randn(B, T, H, D, dtype=dtype)
    k = F.normalize(torch.randn(B, T * num_householder, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T * num_householder, H, D, dtype=dtype)
    beta = torch.rand(B, T * num_householder, H, dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    g = g / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0))

    tri, tri_ht = chunk_gated_delta_product(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        num_householder=num_householder,
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )
    do = torch.randn_like(q)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = chunk_gated_delta_product_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        num_householder=num_householder,
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    if gate_logit_normalizer >= 1 and ref_dg.norm() > 0.01:
        assert_close('dg', ref_dg, tri_dg, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('H', 'D', 'num_householder', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-num_householder{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 64, 3, 0, [0, 63], torch.float16),
            (2, 100, 2, 0, [0, 63, 100, 500, 1000], torch.float16),
            (2, 100, 2, 0, [0, 100, 256, 512, 1500, 1500], torch.float16),
            (2, 128, 2, 0, [0, 100, 300, 800, 1500, 2000], torch.float16),
            (2, 128, 2, 0.5, [0, 31, 111, 799, 1000, 1500, 1800, 2000], torch.float16),
            (2, 128, 2, 0.5, [0, 63, 300, 800, 1000, 1399, 2048], torch.float16),
            (2, 256, 3, 0, [0, 100, 123, 300, 500, 800, 1000, 1500, 2048], torch.float16),
        ]
    ]
)
def test_chunk_varlen(
    H: int,
    D: int,
    num_householder: int,
    mask_p: float,
    cu_seqlens: List[int],
    dtype: torch.dtype,
):
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.nn.functional.normalize(torch.randn((1, T, H, D), dtype=dtype), dim=-1, p=2)
    k = torch.nn.functional.normalize(torch.randn(1, T*num_householder, H, D, dtype=dtype), dim=-1, p=2)
    v = torch.randn((1, T*num_householder, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T*num_householder, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    do = torch.randn_like(q)
    dht = torch.rand_like(h0)
    scale = D ** -0.5

    tri, tri_ht = chunk_gated_delta_product(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        num_householder=num_householder,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref, ref_ht = chunk_gated_delta_product_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        output_final_state=True,
        num_householder=num_householder,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    torch_ref = torch.zeros_like(ref)
    torch_ref_ht = torch.zeros_like(ref_ht)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        q_i = q[:, start:end, :, :]
        k_i = k[:, start*num_householder:end*num_householder, :, :]
        v_i = v[:, start*num_householder:end*num_householder, :, :]
        g_i = g[:, start:end, :]
        beta_i = beta[:, start*num_householder:end*num_householder, :]
        o3_i, h3_i = naive_recurrent_gated_delta_product(
            q_i, k_i, v_i, g_i, beta_i, scale=scale, cu_seqlens=None, output_final_state=True, num_householder=num_householder
        )
        torch_ref[:, start:end, :, :] = o3_i
        torch_ref_ht[i, :, :, :] = h3_i.squeeze(0)

    ((torch_ref * do).sum() + (torch_ref_ht * dht).sum()).backward(retain_graph=True)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
    assert_close('dg', ref_dg, tri_dg, 0.015)
