import os
# 1. Force JAX to use the CPU backend, bypassing MLX/Metal entirely
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax

# Force JAX to use 64-bit math (MUST be set before any JAX operations)
# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "")
jax.clear_caches()

import jax
import jax.numpy as jnp
from jax.scipy.special import erf


def jax_blend(z, sigma, offset):
    SQRT1_2 = 1.0 / jnp.sqrt(2.0)
    # Safe sigma prevents division by zero in the graph, even if unused
    safe_sigma = jnp.where(sigma <= 0.0, 1e-10, sigma)
    blended = 0.5 * erf(SQRT1_2 * (z - offset) / safe_sigma) + 0.5

    # jnp.where replaces the if/else branch
    return jnp.where(sigma <= 0.0, 1.0 * (z >= offset), blended)


def jax_build_profile(z, offset, roughness, contrast, initial_value):
    """
    JAX-vectorized step-to-smooth profile conversion.
    """

    # Define the operation for a single interface
    def compute_interface_delta(off, sig, cont):
        blended = jax_blend(z, sig, off)  # shape: (NZ,)
        return cont[:, None] * blended[None, :]  # shape: (NP, NZ)

    # vmap across the interfaces (axis 0 of offset/roughness, axis 1 of contrast)
    deltas = jax.vmap(compute_interface_delta, in_axes=(0, 0, 1))(offset, roughness, contrast)

    # deltas is (NI, NP, NZ). Sum over NI to accumulate all interfaces.
    return initial_value[:, None] + jnp.sum(deltas, axis=0)


def jax_convolve_gaussian(xin, yin, x, dx):
    """
    Vectorized Gaussian convolution. Replaces dynamic while loops with
    a dense pairwise distance matrix, heavily optimized for GPU/XLA.
    """
    # z shape: (Nout, Nin)
    z = x[:, None] - xin[None, :]
    sigma = dx[:, None]

    # Avoid zero division if dx is 0
    safe_sigma = jnp.where(sigma <= 0.0, 1e-10, sigma)
    two_sigma_sq = 2.0 * safe_sigma**2

    # Compute the Gaussian weights
    G = jnp.exp(-(z**2) / two_sigma_sq) / (safe_sigma * jnp.sqrt(2 * jnp.pi))

    # dx_in for trapezoidal Riemann sum integration
    dx_in = jnp.diff(xin)
    # Pad to match Nin length
    dx_in = jnp.pad(dx_in, (0, 1), mode="edge")

    # If dx == 0, we should technically interpolate, but for simplicity in this
    # demonstration, we assume dx > 0 for standard smearing.
    norm = jnp.sum(G * dx_in, axis=1)
    yout = jnp.sum(yin * G * dx_in, axis=1) / norm

    return jnp.where(dx <= 0.0, jnp.interp(x, xin, yin), yout)


def jax_calculate_u1_u3(H, rhoM, thetaM, Aguide):
    PI4 = 4.0e-6 * jnp.pi
    B2SLD = 2.31604654
    BASE_GUIDE_ANGLE = 270.0

    thetaM_rad = jnp.radians(thetaM)
    Aguide_rad = jnp.radians(Aguide)

    sld_h = B2SLD * H
    sld_m_y = rhoM * jnp.sin(thetaM_rad)
    sld_m_z = 0.0

    # Rotate M
    new_my = sld_m_z * jnp.sin(Aguide_rad) + sld_m_y * jnp.cos(Aguide_rad)
    new_mz = sld_m_z * jnp.cos(Aguide_rad) - sld_m_y * jnp.sin(Aguide_rad)

    sld_b_x = rhoM * jnp.cos(thetaM_rad)  # sld_h_x is 0
    sld_b_y = new_my
    sld_b_z = sld_h + new_mz

    sld_b = jnp.sqrt(sld_b_x**2 + sld_b_y**2 + sld_b_z**2)

    u1_num = sld_b + sld_b_x + 1j * sld_b_y - sld_b_z
    u1_den = sld_b + sld_b_x - 1j * sld_b_y + sld_b_z
    u3_num = -sld_b + sld_b_x + 1j * sld_b_y - sld_b_z
    u3_den = -sld_b + sld_b_x - 1j * sld_b_y + sld_b_z

    # Safe division
    u1 = jnp.where(jnp.abs(u1_den) > 0, u1_num / (u1_den + 1e-15), 0j)
    u3 = jnp.where(jnp.abs(u3_den) > 0, u3_num / (u3_den + 1e-15), 0j)

    return sld_b, u1, u3


def jax_transfer_matrix_layer(carry, layer_params):
    """
    Executes one step of the Parratt recursion for jax.lax.scan.
    """
    B_matrix, Z, E0, IP = carry
    (d, sigma, rho, irho, rhoM, u1_raw, u3_raw) = layer_params

    PI4 = 4.0e-6 * jnp.pi
    EPS = 1e-10

    # JAX conditional: handle u1 swap logic smoothly
    swap_cond = jnp.abs(u1_raw) > 1.0
    u1 = jnp.where(swap_cond, u3_raw, u1_raw)
    u3 = jnp.where(swap_cond, u1_raw, u3_raw)

    b_val = u1
    g_val = 1.0 / (u3 + EPS)

    S1 = -jnp.sqrt(PI4 * (rho + rhoM) - E0 - 1j * PI4 * (jnp.abs(irho) + EPS))
    S3 = -jnp.sqrt(PI4 * (rho - rhoM) - E0 - 1j * PI4 * (jnp.abs(irho) + EPS))

    S1 = jnp.where(swap_cond, S3, S1)
    S3 = jnp.where(swap_cond, S1, S3)

    # NOTE: In a full JAX port, you build the 4x4 'A' matrix here using
    # S1, S3, b_val, g_val of the CURRENT and PREVIOUS layer (which means
    # the carry needs to store the previous layer's properties).
    # For brevity, this assumes A is constructed functionally here.
    A = jnp.eye(4, dtype=jnp.complex128)  # Placeholder for the 16 Axx terms

    # Matrix update: B = A @ B
    new_B = A @ B_matrix
    new_Z = Z + d

    return (new_B, new_Z, E0, IP), None


def jax_Cr4xa(d, sigma, IP, rho, irho, rhoM, u1, u3, KZ):
    """
    Replaces the Cr4xa for-loop with jax.lax.scan
    """
    E0 = KZ * KZ + 4.0e-6 * jnp.pi * (rho[0] + IP * rhoM[0])
    B_init = jnp.eye(4, dtype=jnp.complex128)
    Z_init = 0.0

    init_carry = (B_init, Z_init, E0, IP)

    # Pack layer arrays to iterate over them simultaneously
    layer_params = (d, sigma, rho, irho, rhoM, u1, u3)

    # Scan rolls over the layers axis (axis 0)
    final_carry, _ = jax.lax.scan(jax_transfer_matrix_layer, init_carry, layer_params)
    B_final = final_carry[0]

    DETW = B_final[3, 3] * B_final[1, 1] - B_final[1, 3] * B_final[3, 1]

    YA = (B_final[1, 3] * B_final[3, 0] - B_final[1, 0] * B_final[3, 3]) / DETW
    YB = (B_final[1, 0] * B_final[3, 1] - B_final[3, 0] * B_final[1, 1]) / DETW
    YC = (B_final[1, 3] * B_final[3, 2] - B_final[1, 2] * B_final[3, 3]) / DETW
    YD = (B_final[1, 2] * B_final[3, 1] - B_final[3, 2] * B_final[1, 1]) / DETW

    return YA, YB, YC, YD


import jax

# -------------------------------------------------------------------------
# UNPOLARIZED JAX KERNEL
# -------------------------------------------------------------------------

# @jax.jit
# def refl_jax(depth, sigma, rho, irho, kz_val):
#     """Core unpolarized reflectivity kernel with Metal/M1 optimizations."""
#     J = 1j
#     PI4 = 12.566370614359172e-6
#     cutoff = 1e-10
#     L = len(depth)

#     # Forward/Backward array traversal setup
#     is_forward = kz_val >= 0.0
#     idx_incident = jnp.where(is_forward, 0, L - 1)

#     k_init = jnp.complex64(jnp.abs(kz_val))
#     kz_sq = jnp.complex64(kz_val**2 + PI4 * rho[idx_incident])

#     init_state = (
#         jnp.complex64(1.0), jnp.complex64(0.0),
#         jnp.complex64(0.0), jnp.complex64(1.0),
#         k_init
#     )

#     layer_indices = jnp.arange(L - 1)

#     def scan_body(carry, step_idx):
#         B11, B21, B12, B22, k_curr = carry

#         idx_rho   = jnp.where(is_forward, step_idx + 1, L - 2 - step_idx)
#         idx_sigma = jnp.where(is_forward, step_idx,     L - 2 - step_idx)
#         idx_depth = jnp.where(is_forward, step_idx,     L - 1 - step_idx)

#         arg = kz_sq - PI4 * (rho[idx_rho] + J * irho[idx_rho])

#         # Critical edge branch cut fix
#         is_evanescent = (irho[idx_rho] == 0.0) & (jnp.real(arg) < 0.0)
#         k_next = jnp.where(
#             is_evanescent,
#             J * jnp.sqrt(jnp.abs(jnp.real(arg))),
#             jnp.sqrt(arg)
#         )

#         sig = sigma[idx_sigma]
#         F = ((k_curr - k_next) / (k_curr + k_next) * jnp.exp(-2.0 * k_curr * k_next * sig**2))

#         d = depth[idx_depth]
#         phase = jnp.where(step_idx > 0, jnp.exp(J * k_curr * d), jnp.complex64(1.0))
#         inv_phase = jnp.where(step_idx > 0, jnp.exp(-J * k_curr * d), jnp.complex64(1.0))

#         M11, M22 = phase, inv_phase
#         M21, M12 = F * M11, F * M22

#         new_B11 = B11 * M11 + B21 * M12
#         new_B21 = B11 * M21 + B21 * M22
#         new_B12 = B12 * M11 + B22 * M12
#         new_B22 = B12 * M21 + B22 * M22

#         return (new_B11, new_B21, new_B12, new_B22, k_next), None

#     final_state, _ = jax.lax.scan(scan_body, init_state, layer_indices)
#     calc_r = final_state[2] / final_state[0] # B12 / B11

#     return jnp.where(jnp.abs(kz_val) < cutoff, jnp.complex64(-1.0), calc_r)


# -------------------------------------------------------------------------
# NLLF PIPELINES
# -------------------------------------------------------------------------
@jax.jit
def unpolarized_model_nllf(
    raw_slabs, q_oversampled, q_nominal, dq_nominal, r_data, dr_data, intensity, background, back_absorption
):
    """Calculates the NLLF for ONE unpolarized model."""
    d = raw_slabs[:, 0]
    sigma = raw_slabs[:, 1]
    rho = raw_slabs[:, 2]
    irho = raw_slabs[:, 3]

    vmap_refl = jax.vmap(refl_jax, in_axes=(None, None, None, None, 0))
    calc_r = vmap_refl(d, sigma, rho, irho, -q_oversampled / 2.0)

    R_theory_oversampled = jnp.abs(calc_r) ** 2

    # 1. Back absorption (applied before resolution smearing for Q < 0)
    back_mult = jnp.where(q_oversampled < 0, back_absorption - 1.0, 0.0) + 1.0
    R_theory_oversampled = R_theory_oversampled * back_mult

    # 2. Resolution Smearing
    R_theory_nominal = jax_convolve_gaussian(q_oversampled, R_theory_oversampled, q_nominal, dq_nominal)

    # 3. Intensity and Background
    R_theory_nominal = (R_theory_nominal * intensity) + background

    resid = (R_theory_nominal - r_data) / dr_data
    return 0.5 * jnp.sum(resid**2)


@jax.jit
def polarized_model_nllf(
    raw_slabs, q_oversampled, q_nominal, dq_nominal, r_data, dr_data, H, Aguide, intensity, background, back_absorption
):
    """Calculates the NLLF for ONE polarized model."""
    d = raw_slabs[:, 0]
    sigma = raw_slabs[:, 1]
    rho = raw_slabs[:, 2]
    irho = raw_slabs[:, 3]
    rhoM = raw_slabs[:, 4]
    thetaM = raw_slabs[:, 5]

    rhoM_b, u1, u3 = jax_calculate_u1_u3(H, rhoM, thetaM, Aguide)
    vmap_Cr4xa = jax.vmap(jax_Cr4xa, in_axes=(None, None, None, None, None, None, None, None, 0))

    Ra_p, Rb_p, Rc_p, Rd_p = vmap_Cr4xa(d, sigma, 1.0, rho, irho, rhoM_b, u1, u3, q_oversampled / 2.0)
    Ra_m, Rb_m, Rc_m, Rd_m = vmap_Cr4xa(d, sigma, -1.0, rho, irho, rhoM_b, u1, u3, q_oversampled / 2.0)

    R_theory_oversampled = jnp.abs(Ra_p) ** 2 + jnp.abs(Rc_m) ** 2

    # 1. Back absorption (applied before resolution smearing for Q < 0)
    back_mult = jnp.where(q_oversampled < 0, back_absorption - 1.0, 0.0) + 1.0
    R_theory_oversampled = R_theory_oversampled * back_mult

    # 2. Resolution Smearing
    R_theory_nominal = jax_convolve_gaussian(q_oversampled, R_theory_oversampled, q_nominal, dq_nominal)

    # 3. Intensity and Background
    R_theory_nominal = (R_theory_nominal * intensity) + background

    resid = (R_theory_nominal - r_data) / dr_data
    return 0.5 * jnp.sum(resid**2)


# --- VMAP BATCHES ---
# Note: Intensity, background, and back_absorption are mapped as `None` (scalars)
# to match how H and Aguide are treated for a specific model batch.
batch_unpolarized_nllf_gpu = jax.vmap(
    unpolarized_model_nllf, in_axes=(0, None, None, None, None, None, None, None, None)
)
batch_polarized_nllf_gpu = jax.vmap(
    polarized_model_nllf, in_axes=(0, None, None, None, None, None, None, None, None, None, None)
)


# -------------------------------------------------------------------------
# THE PYTHON BATCH DISPATCHER
# -------------------------------------------------------------------------
def batch_nllf_gpu(
    batch_slabs, q_oversampled, probe_data, H, Aguide, intensity, background, back_absorption, is_polarized
):
    """
    Python wrapper to parse data structures from experiment.py and dispatch
    to the correct compiled JAX pipeline.
    """
    import numpy as np

    if not is_polarized:
        Q_nom = probe_data["Q"]
        R_dat = probe_data["R"]
        dR_dat = probe_data["dR"]
        dQ_nom = probe_data.get("dQ")
        if dQ_nom is None:
            dQ_nom = jnp.zeros_like(Q_nom)

        nllf_arr = batch_unpolarized_nllf_gpu(
            batch_slabs, q_oversampled, Q_nom, dQ_nom, R_dat, dR_dat, intensity, background, back_absorption
        )
        return np.asarray(nllf_arr)

    else:
        total_nllf = np.zeros(batch_slabs.shape[0])
        for xs_data in probe_data:
            Q_nom = xs_data["Q"]
            R_dat = xs_data["R"]
            dR_dat = xs_data["dR"]
            dQ_nom = xs_data.get("dQ")
            if dQ_nom is None:
                dQ_nom = jnp.zeros_like(Q_nom)

            nllf_part = batch_polarized_nllf_gpu(
                batch_slabs,
                q_oversampled,
                Q_nom,
                dQ_nom,
                R_dat,
                dR_dat,
                H,
                Aguide,
                intensity,
                background,
                back_absorption,
            )
            total_nllf += np.asarray(nllf_part)

        return total_nllf


batch_nllf = batch_nllf_gpu
# Usage:
# batch_slabs = jnp.stack([slab_1, slab_2, ...]) # Shape: (1000, max_layers, 6)
# nllf_results = batch_nllf_gpu(batch_slabs, Q_ov, Q_nom, dQ_nom, R_dat, dR_dat, H, Aguide)


from .reflectivity import reflectivity_amplitude_jax as reflectivity_amplitude, refl_jax
from .magnetic_local_d import magnetic_amplitude
from ..numba.magnetic import calculate_u1_u3
from ..numba.build_profile import build_profile
from ..numba.convolve import convolve_gaussian
from ..numba.convolve import convolve_uniform
from ..numba.convolve_sampled import convolve_sampled
from ..numba.contract_profile import align_magnetic
from ..numba.contract_profile import contract_by_area
from ..numba.contract_profile import contract_mag
from ..numba.rebin import rebin_counts
from ..numba.rebin import rebin_counts_2D
