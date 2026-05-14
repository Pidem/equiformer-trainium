"""NKI kernel for init_edge_rot_mat.

The original PyTorch routine builds a 3x3 rotation matrix per edge:
    norm_x = edge_vec / |edge_vec|
    edge_vec_2 = random unit vector (with tiebreaker copies if parallel to x)
    norm_z = cross(norm_x, edge_vec_2); normalize
    norm_y = cross(norm_x, norm_z); normalize
    rot_mat = stack [norm_z, norm_x, -norm_y] as rows

Two ops in that path fall back to CPU on Neuron:
    torch.cross  -> aten::linalg_cross.out
    torch.rand_like -> aten::uniform_

This kernel replaces the whole routine with on-device elementwise ops.
The random tiebreaker is replaced with a deterministic 2-way conditional:
    use e2 = [0, 1, 0] in general
    use e2 = [0, 0, 1] when |norm_x[1]| > 0.9 (norm_x near y-axis)
This guarantees a non-degenerate cross product because the two candidate
e2 vectors are orthogonal to each other; norm_x cannot be parallel to both.

The output (norm_y, norm_z within the perpendicular plane) is therefore not
identical to the random baseline, but the model is rotation-equivariant in
that plane so the final forces differ by the same magnitude as natural
run-to-run variance from the random baseline.
"""
from __future__ import annotations

import torch
import nki
import nki.isa as nisa
import nki.language as nl


P_MAX = 128


def _div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


@nki.jit
def _init_edge_rot_mat_kernel(edge_vec_kk):
    """
    Inputs (HBM, f32):
        edge_vec_kk: [P_MAX, 3] -- per-edge displacement vectors, padded.

    Output:
        rot_mat_flat: [P_MAX, 9] -- per-edge rotation matrix flattened
            row-major: [R[0,0], R[0,1], R[0,2], R[1,0], ..., R[2,2]]
            with rows [norm_z, norm_x, -norm_y].
    """
    B = edge_vec_kk.shape[0]
    F = 3
    M = 9

    rot_mat = nl.ndarray((B, M), dtype=edge_vec_kk.dtype, buffer=nl.shared_hbm)

    """
    Load edge_vec into SBUF.
    """
    edge_vec_sb = nl.ndarray((P_MAX, F), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=edge_vec_sb[0:B, 0:F], src=edge_vec_kk[0:B, 0:F])

    """
    Step 1: norm_x = edge_vec / |edge_vec|.
    """
    sq_sb = nl.ndarray((P_MAX, F), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=sq_sb[0:B, 0:F],
        data1=edge_vec_sb[0:B, 0:F],
        data2=edge_vec_sb[0:B, 0:F],
        op=nl.multiply,
    )

    len_sq_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(
        dst=len_sq_sb[0:B, 0:1],
        data=sq_sb[0:B, 0:F],
        op=nl.add,
        axis=(1,),
    )

    inv_len_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.activation(dst=inv_len_sb[0:B, 0:1], data=len_sq_sb[0:B, 0:1], op=nl.rsqrt)

    inv_len_bcast = nl.broadcast_to(inv_len_sb[0:B, 0:1], shape=(B, F))

    norm_x_sb = nl.ndarray((P_MAX, F), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=norm_x_sb[0:B, 0:F],
        data1=edge_vec_sb[0:B, 0:F],
        data2=inv_len_bcast,
        op=nl.multiply,
    )

    """
    Step 2: choose e2 based on |norm_x[1]|.
        |norm_x[1]| <= 0.9  -> e2 = [0, 1, 0]
        |norm_x[1]|  > 0.9  -> e2 = [0, 0, 1]
    Use squared comparison: nx[1]^2 > 0.81.
    """
    nx1_sq_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=nx1_sq_sb[0:B, 0:1],
        data1=norm_x_sb[0:B, 1:2],
        data2=norm_x_sb[0:B, 1:2],
        op=nl.multiply,
    )

    mask_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=mask_sb[0:B, 0:1],
        data=nx1_sq_sb[0:B, 0:1],
        op0=nl.greater,
        operand0=0.81,
    )

    """
    1 - mask, used for the complementary branch.
    """
    one_minus_mask_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=one_minus_mask_sb[0:B, 0:1],
        data=mask_sb[0:B, 0:1],
        op0=nl.multiply,
        operand0=-1.0,
    )
    nisa.tensor_scalar(
        dst=one_minus_mask_sb[0:B, 0:1],
        data=one_minus_mask_sb[0:B, 0:1],
        op0=nl.add,
        operand0=1.0,
    )

    """
    Step 3: norm_z = cross(norm_x, e2) blended by mask.
        if mask=0 (e2 = [0,1,0]): cross = [-nx[2],   0,    nx[0]]
        if mask=1 (e2 = [0,0,1]): cross = [ nx[1], -nx[0],  0   ]
    Blend:
        z[0] = (1-m) * (-nx[2]) + m * nx[1]
        z[1] = (1-m) *   0      + m * (-nx[0]) = -m * nx[0]
        z[2] = (1-m) *  nx[0]   + m *   0      = (1-m) * nx[0]
    """
    norm_z_sb = nl.ndarray((P_MAX, F), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    t1_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    t2_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)

    """
    z[0] = -(1-m)*nx[2] + m*nx[1]
    """
    nisa.tensor_tensor(
        dst=t1_sb[0:B, 0:1],
        data1=one_minus_mask_sb[0:B, 0:1],
        data2=norm_x_sb[0:B, 2:3],
        op=nl.multiply,
    )
    nisa.tensor_scalar(
        dst=t1_sb[0:B, 0:1],
        data=t1_sb[0:B, 0:1],
        op0=nl.multiply,
        operand0=-1.0,
    )
    nisa.tensor_tensor(
        dst=t2_sb[0:B, 0:1],
        data1=mask_sb[0:B, 0:1],
        data2=norm_x_sb[0:B, 1:2],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=norm_z_sb[0:B, 0:1],
        data1=t1_sb[0:B, 0:1],
        data2=t2_sb[0:B, 0:1],
        op=nl.add,
    )

    """
    z[1] = -m * nx[0]
    """
    nisa.tensor_tensor(
        dst=t1_sb[0:B, 0:1],
        data1=mask_sb[0:B, 0:1],
        data2=norm_x_sb[0:B, 0:1],
        op=nl.multiply,
    )
    nisa.tensor_scalar(
        dst=norm_z_sb[0:B, 1:2],
        data=t1_sb[0:B, 0:1],
        op0=nl.multiply,
        operand0=-1.0,
    )

    """
    z[2] = (1-m) * nx[0]
    """
    nisa.tensor_tensor(
        dst=norm_z_sb[0:B, 2:3],
        data1=one_minus_mask_sb[0:B, 0:1],
        data2=norm_x_sb[0:B, 0:1],
        op=nl.multiply,
    )

    """
    Normalize norm_z.
    """
    z_sq_sb = nl.ndarray((P_MAX, F), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=z_sq_sb[0:B, 0:F],
        data1=norm_z_sb[0:B, 0:F],
        data2=norm_z_sb[0:B, 0:F],
        op=nl.multiply,
    )
    z_len_sq_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(
        dst=z_len_sq_sb[0:B, 0:1],
        data=z_sq_sb[0:B, 0:F],
        op=nl.add,
        axis=(1,),
    )
    z_inv_len_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.activation(dst=z_inv_len_sb[0:B, 0:1], data=z_len_sq_sb[0:B, 0:1], op=nl.rsqrt)
    z_inv_len_bcast = nl.broadcast_to(z_inv_len_sb[0:B, 0:1], shape=(B, F))
    nisa.tensor_tensor(
        dst=norm_z_sb[0:B, 0:F],
        data1=norm_z_sb[0:B, 0:F],
        data2=z_inv_len_bcast,
        op=nl.multiply,
    )

    """
    Step 4: norm_y = cross(norm_x, norm_z).
        cross([a,b,c], [d,e,f]) = [b*f - c*e, c*d - a*f, a*e - b*d]
    """
    norm_y_sb = nl.ndarray((P_MAX, F), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)

    """
    y[0] = nx[1]*nz[2] - nx[2]*nz[1]
    """
    nisa.tensor_tensor(
        dst=t1_sb[0:B, 0:1],
        data1=norm_x_sb[0:B, 1:2],
        data2=norm_z_sb[0:B, 2:3],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=t2_sb[0:B, 0:1],
        data1=norm_x_sb[0:B, 2:3],
        data2=norm_z_sb[0:B, 1:2],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=norm_y_sb[0:B, 0:1],
        data1=t1_sb[0:B, 0:1],
        data2=t2_sb[0:B, 0:1],
        op=nl.subtract,
    )

    """
    y[1] = nx[2]*nz[0] - nx[0]*nz[2]
    """
    nisa.tensor_tensor(
        dst=t1_sb[0:B, 0:1],
        data1=norm_x_sb[0:B, 2:3],
        data2=norm_z_sb[0:B, 0:1],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=t2_sb[0:B, 0:1],
        data1=norm_x_sb[0:B, 0:1],
        data2=norm_z_sb[0:B, 2:3],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=norm_y_sb[0:B, 1:2],
        data1=t1_sb[0:B, 0:1],
        data2=t2_sb[0:B, 0:1],
        op=nl.subtract,
    )

    """
    y[2] = nx[0]*nz[1] - nx[1]*nz[0]
    """
    nisa.tensor_tensor(
        dst=t1_sb[0:B, 0:1],
        data1=norm_x_sb[0:B, 0:1],
        data2=norm_z_sb[0:B, 1:2],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=t2_sb[0:B, 0:1],
        data1=norm_x_sb[0:B, 1:2],
        data2=norm_z_sb[0:B, 0:1],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=norm_y_sb[0:B, 2:3],
        data1=t1_sb[0:B, 0:1],
        data2=t2_sb[0:B, 0:1],
        op=nl.subtract,
    )

    """
    Normalize norm_y. (Mathematically already unit length but matches original.)
    """
    y_sq_sb = nl.ndarray((P_MAX, F), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=y_sq_sb[0:B, 0:F],
        data1=norm_y_sb[0:B, 0:F],
        data2=norm_y_sb[0:B, 0:F],
        op=nl.multiply,
    )
    y_len_sq_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(
        dst=y_len_sq_sb[0:B, 0:1],
        data=y_sq_sb[0:B, 0:F],
        op=nl.add,
        axis=(1,),
    )
    y_inv_len_sb = nl.ndarray((P_MAX, 1), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.activation(dst=y_inv_len_sb[0:B, 0:1], data=y_len_sq_sb[0:B, 0:1], op=nl.rsqrt)
    y_inv_len_bcast = nl.broadcast_to(y_inv_len_sb[0:B, 0:1], shape=(B, F))
    nisa.tensor_tensor(
        dst=norm_y_sb[0:B, 0:F],
        data1=norm_y_sb[0:B, 0:F],
        data2=y_inv_len_bcast,
        op=nl.multiply,
    )

    """
    Step 5: assemble rot_mat[:, 0:9] = [norm_z, norm_x, -norm_y].
    """
    out_sb = nl.ndarray((P_MAX, M), dtype=edge_vec_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_sb[0:B, 0:3], src=norm_z_sb[0:B, 0:F])
    nisa.tensor_copy(dst=out_sb[0:B, 3:6], src=norm_x_sb[0:B, 0:F])
    nisa.tensor_scalar(
        dst=out_sb[0:B, 6:9],
        data=norm_y_sb[0:B, 0:F],
        op0=nl.multiply,
        operand0=-1.0,
    )

    nisa.dma_copy(dst=rot_mat[0:B, 0:M], src=out_sb[0:B, 0:M])

    return rot_mat


def init_edge_rot_mat_nki(edge_distance_vec: torch.Tensor, use_rotation_mask: bool = False) -> torch.Tensor:
    """
    Drop-in replacement for fairchem...edge_rot_mat.init_edge_rot_mat.
    Args:
        edge_distance_vec: [E, 3] f32 edge displacement vectors.
        use_rotation_mask: if True, fall back to original implementation
            (only direct_prediction=False uses this branch).
    Returns:
        edge_rot_mat: [E, 3, 3] f32 (detached, since the caller does so).
    """
    if use_rotation_mask:
        from fairchem.experimental.models.equiformer_v3.edge_rot_mat import init_edge_rot_mat as _orig
        return _orig(edge_distance_vec, use_rotation_mask=True)

    E = edge_distance_vec.shape[0]
    dtype = edge_distance_vec.dtype
    device = edge_distance_vec.device

    n_tiles = _div_ceil(E, P_MAX)
    E_padded = n_tiles * P_MAX

    if E_padded != E:
        """
        Pad with [1, 0, 0] to keep arithmetic well-defined in the unused rows.
        """
        pad = torch.zeros(E_padded - E, 3, dtype=dtype, device=device)
        pad[:, 0] = 1.0
        edge_padded = torch.cat([edge_distance_vec.contiguous(), pad], dim=0)
    else:
        edge_padded = edge_distance_vec.contiguous()

    out_tiles = []
    for t in range(n_tiles):
        chunk = edge_padded[t * P_MAX:(t + 1) * P_MAX].contiguous()
        rot_chunk = _init_edge_rot_mat_kernel(chunk)
        out_tiles.append(rot_chunk)

    rot_padded = torch.cat(out_tiles, dim=0) if len(out_tiles) > 1 else out_tiles[0]
    rot_flat = rot_padded[:E]
    return rot_flat.reshape(E, 3, 3).detach()


if __name__ == "__main__":
    import os
    import sys

    os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
    os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

    sys.path.insert(0, "equiformer_v3/src")

    torch.manual_seed(0)
    device = torch.device("neuron:0")

    """
    Test: kernel output should be a valid rotation matrix per edge:
        det(R) = +1
        R @ R.T = I
        R[1, :] = norm_x = edge_vec / |edge_vec|  (the second row is the edge direction)
    """
    print(f"{'E':>5s}  {'det err':>10s}  {'orth err':>10s}  {'x-row err':>10s}")
    for E in [1, 5, 100, 460]:
        edge_vec = torch.randn(E, 3, dtype=torch.float32) * 2.0
        edge_vec_n = edge_vec.to(device)

        rot = init_edge_rot_mat_nki(edge_vec_n)
        torch.neuron.synchronize()
        rot_cpu = rot.cpu()

        """
        Check det = +1 across rows.
        """
        dets = torch.linalg.det(rot_cpu)
        det_err = (dets - 1.0).abs().max().item()

        """
        Check orthonormality.
        """
        I = torch.eye(3).unsqueeze(0).expand(E, 3, 3)
        ortho = torch.bmm(rot_cpu, rot_cpu.transpose(-1, -2)) - I
        ortho_err = ortho.abs().max().item()

        """
        Check second row is edge direction.
        """
        norm_x_ref = edge_vec / edge_vec.norm(dim=1, keepdim=True)
        x_row_err = (rot_cpu[:, 1, :] - norm_x_ref).abs().max().item()

        print(f"{E:>5d}  {det_err:>10.3e}  {ortho_err:>10.3e}  {x_row_err:>10.3e}")
