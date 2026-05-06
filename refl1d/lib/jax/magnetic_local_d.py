import jax

# Explicitly enforce 32-bit mode
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np


# --- HELPER: Force Euler Decomposition for ALL complex exponentials ---
def safe_complex_exp(z):
    mag = jnp.exp(jnp.real(z))
    phase = jnp.imag(z)
    return mag * (jnp.cos(phase) + 1j * jnp.sin(phase))


import jax
import jax.numpy as jnp


def scan_step_independent(carry, layer_data):
    # Unpack the carry (flat B matrix and previous layer states)
    B_flat, S1L, S3L, BL, GL = carry

    # Unpack layer-specific data for the 'Next' layer (LP) and current layer (L)
    # E0 is passed here to ensure it is treated as a dynamic input for the kernel
    dL, rhoLP, rhoMLP, irhoLP, u1LP, u3LP, sigmaL, E0 = layer_data

    EPS = 1e-7
    PI4 = 4.0e-6 * jnp.pi

    # 3. Calculate S1LP and S3LP directly (Replaces get_S and get_S3)
    # This inline math is the most stable way to present logic to the Metal compiler
    S1LP_raw = -jnp.sqrt(PI4 * (rhoLP + rhoMLP) - E0 - 1j * PI4 * (jnp.abs(irhoLP) + EPS))
    S3LP_raw = -jnp.sqrt(PI4 * (rhoLP - rhoMLP) - E0 - 1j * PI4 * (jnp.abs(irhoLP) + EPS))

    # 4. Handle Magnetic Flip Conditions
    flip_cond = jnp.abs(u1LP) > 1.0
    S1LP = jnp.where(flip_cond, S3LP_raw, S1LP_raw)
    S3LP = jnp.where(flip_cond, S1LP_raw, S3LP_raw)

    BLP = jnp.where(flip_cond, u3LP, u1LP)
    GLP = jnp.where(flip_cond, 1.0 / (u1LP + EPS), 1.0 / (u3LP + EPS))

    # --- Transfer Matrix Components (A matrix) ---
    DELTA = 0.5 / (1.0 - (BLP * GLP))
    DBB, DBG = (BL - BLP) * DELTA, (1.0 - BL * GLP) * DELTA
    DGB, DGG = (1.0 - GL * BLP) * DELTA, (GL - GLP) * DELTA

    # Thickness propagation via safe Euler decomposition
    ES1L = safe_complex_exp(S1L * dL)
    ENS1L = safe_complex_exp(-S1L * dL)
    ES3L = safe_complex_exp(S3L * dL)
    ENS3L = safe_complex_exp(-S3L * dL)

    FS1S1, FS1S3 = S1L / S1LP, S1L / S3LP
    FS3S1, FS3S3 = S3L / S1LP, S3L / S3LP

    # Roughness factors (NC) - set to 1.0 as per your testing logic
    nc_11 = nc_13 = nc_31 = nc_33 = 1.0 + 0j

    # Define A-matrix elements (Row-major: A11=Row0,Col0; A12=Row0,Col1...)
    A11, A12 = DBG * (1.0 + FS1S1) * ES1L, DBG * (1.0 - FS1S1) * nc_11 * ENS1L
    A13, A14 = DGG * (1.0 + FS3S1) * ES3L, DGG * (1.0 - FS3S1) * nc_31 * ENS3L

    A21, A22 = DBG * (1.0 - FS1S1) * nc_11 * ES1L, DBG * (1.0 + FS1S1) * ENS1L
    A23, A24 = DGG * (1.0 - FS3S1) * nc_31 * ES3L, DGG * (1.0 + FS3S1) * ENS3L

    A31, A32 = DBB * (1.0 + FS1S3) * ES1L, DBB * (1.0 - FS1S3) * nc_13 * ENS1L
    A33, A34 = DGB * (1.0 + FS3S3) * ES3L, DGB * (1.0 - FS3S3) * nc_33 * ENS3L

    A41, A42 = DBB * (1.0 - FS1S3) * nc_13 * ES1L, DBB * (1.0 + FS1S3) * ENS1L
    A43, A44 = DGB * (1.0 - FS3S3) * nc_33 * ES3L, DGB * (1.0 + FS3S3) * ENS3L

    # --- Flattened Matrix Multiplication: new_B = A @ B ---
    # Row 0
    nb0 = A11 * B_flat[0] + A12 * B_flat[4] + A13 * B_flat[8] + A14 * B_flat[12]
    nb1 = A11 * B_flat[1] + A12 * B_flat[5] + A13 * B_flat[9] + A14 * B_flat[13]
    nb2 = A11 * B_flat[2] + A12 * B_flat[6] + A13 * B_flat[10] + A14 * B_flat[14]
    nb3 = A11 * B_flat[3] + A12 * B_flat[7] + A13 * B_flat[11] + A14 * B_flat[15]

    # Row 1
    nb4 = A21 * B_flat[0] + A22 * B_flat[4] + A23 * B_flat[8] + A24 * B_flat[12]
    nb5 = A21 * B_flat[1] + A22 * B_flat[5] + A23 * B_flat[9] + A24 * B_flat[13]
    nb6 = A21 * B_flat[2] + A22 * B_flat[6] + A23 * B_flat[10] + A24 * B_flat[14]
    nb7 = A21 * B_flat[3] + A22 * B_flat[7] + A23 * B_flat[11] + A24 * B_flat[15]

    # Row 2
    nb8 = A31 * B_flat[0] + A32 * B_flat[4] + A33 * B_flat[8] + A34 * B_flat[12]
    nb9 = A31 * B_flat[1] + A32 * B_flat[5] + A33 * B_flat[9] + A34 * B_flat[13]
    nb10 = A31 * B_flat[2] + A32 * B_flat[6] + A33 * B_flat[10] + A34 * B_flat[14]
    nb11 = A31 * B_flat[3] + A32 * B_flat[7] + A33 * B_flat[11] + A34 * B_flat[15]

    # Row 3
    nb12 = A41 * B_flat[0] + A42 * B_flat[4] + A43 * B_flat[8] + A44 * B_flat[12]
    nb13 = A41 * B_flat[1] + A42 * B_flat[5] + A43 * B_flat[9] + A44 * B_flat[13]
    nb14 = A41 * B_flat[2] + A42 * B_flat[6] + A43 * B_flat[10] + A44 * B_flat[14]
    nb15 = A41 * B_flat[3] + A42 * B_flat[7] + A43 * B_flat[11] + A44 * B_flat[15]

    # Use jnp.stack to create the new carry flat vector
    # stack() is generally more stable than jnp.array() on MPS
    new_B_flat = jnp.stack([nb0, nb1, nb2, nb3, nb4, nb5, nb6, nb7, nb8, nb9, nb10, nb11, nb12, nb13, nb14, nb15])

    norm = jnp.max(jnp.abs(new_B_flat)) + 1e-7
    new_B_flat = new_B_flat / norm

    return (new_B_flat, S1LP, S3LP, BLP, GLP), None


# @jax.jit
def jax_Cr4xa_single_local(d, sigma, IP, rho, irho, rhoM, u1, u3, kz):
    print(f"DEBUG: Compiling kernel for d_shape={d.shape}, d_dtype={d.dtype}")
    print(f"jax_Cr4xa... {rho[1]}")
    EPS = 1e-7
    PI4 = 4.0e-6 * jnp.pi

    is_forward = kz >= 1e-10

    d_use = jnp.where(is_forward, d, jnp.flip(d))
    rho_use = jnp.where(is_forward, rho, jnp.flip(rho))
    irho_use = jnp.where(is_forward, irho, jnp.flip(irho))
    rhoM_use = jnp.where(is_forward, rhoM, jnp.flip(rhoM))
    u1_use = jnp.where(is_forward, u1, jnp.flip(u1))
    u3_use = jnp.where(is_forward, u3, jnp.flip(u3))
    sigma_use = jnp.where(is_forward, sigma, jnp.flip(sigma))

    E0 = kz**2 + PI4 * (rho_use[0] + IP * rhoM_use[0])

    def get_S(rho_val, rhom_val, irho_val):
        return -jnp.sqrt(PI4 * (rho_val + rhom_val) - E0 - 1j * PI4 * (jnp.abs(irho_val) + EPS))

    def get_S3(rho_val, rhom_val, irho_val):
        return -jnp.sqrt(PI4 * (rho_val - rhom_val) - E0 - 1j * PI4 * (jnp.abs(irho_val) + EPS))

    S1L_init = get_S(rho_use[0], rhoM_use[0], irho_use[0])
    S3L_init = get_S3(rho_use[0], rhoM_use[0], irho_use[0])

    flip_cond_init = jnp.abs(u1_use[0]) > 1.0
    S1L_init, S3L_init = jnp.where(flip_cond_init, S3L_init, S1L_init), jnp.where(flip_cond_init, S1L_init, S3L_init)

    BL_init = 0.0 + 0j
    GL_init = 0.0 + 0j

    # Initialize B as a flat 16-element array
    B_init_flat = jnp.eye(4, dtype=jnp.complex64).reshape(-1)

    # Prepare the data for all N-1 steps
    # d_use[i] and sigma_use[i] are used for current layer propagation
    # rho_use[i+1] etc. are used for the 'Next' layer calculations
    N = d.shape[0]

    # 2. Use that N to create the E0 broadcast array
    # Since lax.scan iterates N-1 times, we need a length of N-1
    # E0_broadcast = jnp.full((N - 1,), E0)
    layer_inputs = (
        d_use[: N - 1],  # d_use[:N-1] has length N-1
        rho_use[1:N],  # rho_use[1:N] has length N-1
        rhoM_use[1:N],  # rhoM_use[1:N] has length N-1
        irho_use[1:N],  # irho_use[1:N] has length N-1
        u1_use[1:N],  # u1_use[1:N] has length N-1
        u3_use[1:N],  # u3_use[1:N] has length N-1
        sigma_use[: N - 1],  # sigma_use[:N-1] has length N-1
        jnp.full((N - 1,), E0),  # E0 for every step, length N-1
    )

    init_carry = (B_init_flat, S1L_init, S3L_init, BL_init, GL_init)
    final_carry, _ = jax.lax.scan(scan_step_independent, init_carry, layer_inputs)

    # Reshape back at the very end to compute DETW
    B_final = final_carry[0].reshape((4, 4))

    DETW = B_final[3, 3] * B_final[1, 1] - B_final[1, 3] * B_final[3, 1]

    YA = jnp.where(IP > 0, (B_final[1, 3] * B_final[3, 0] - B_final[1, 0] * B_final[3, 3]) / DETW, 0.0j)
    YB = jnp.where(IP > 0, (B_final[1, 0] * B_final[3, 1] - B_final[3, 0] * B_final[1, 1]) / DETW, 0.0j)
    YC = (B_final[1, 3] * B_final[3, 2] - B_final[1, 2] * B_final[3, 3]) / DETW
    YD = (B_final[1, 2] * B_final[3, 1] - B_final[3, 2] * B_final[1, 1]) / DETW

    is_valid = jnp.abs(kz) >= 1e-10
    YA = jnp.where(is_valid, YA, -1.0 + 0j)
    YB = jnp.where(is_valid, YB, 0.0 + 0j)
    YC = jnp.where(is_valid, YC, 0.0 + 0j)
    YD = jnp.where(is_valid, YD, -1.0 + 0j)

    return YA, YB, YC, YD


# @jax.jit
def jax_magnetic_amplitude_core(d, sigma, rho, irho, rhoM, u1, u3, KZ):
    # This print MUST trigger only once if the shape stays the same.
    # If the values don't change, JAX has "baked" the data into the shader.
    print(f"DEBUG: JIT Trace triggered for KZ shape {KZ.shape}")

    vmap_fn = jax.vmap(jax_Cr4xa_single_local, in_axes=(None, None, None, None, None, None, None, None, 0))

    Ra_p, Rb_p, Rc_p, Rd_p = vmap_fn(d, sigma, 1.0, rho, irho, rhoM, u1, u3, KZ)
    Ra_m, Rb_m, Rc_m, Rd_m = vmap_fn(d, sigma, -1.0, rho, irho, rhoM, u1, u3, KZ)

    return Ra_p, Rb_p, Rc_m, Rd_m


def magnetic_amplitude(d, sigma, rho, irho, rhoM, u1, u3, KZ, rho_index, Ra, Rb, Rc, Rd):
    # 1. Force conversion to NumPy first to break any previous JAX tracing links
    # 2. Re-wrap as JAX arrays to ensure they are treated as dynamic Traced values
    d_jnp = jnp.array(np.array(d, copy=True), dtype=jnp.float32)
    sigma_jnp = jnp.array(np.array(sigma, copy=True), dtype=jnp.float32)
    rho_jnp = jnp.array(np.array(rho, copy=True), dtype=jnp.float32)
    irho_jnp = jnp.array(np.array(irho, copy=True), dtype=jnp.float32)
    rhoM_jnp = jnp.array(np.array(rhoM, copy=True), dtype=jnp.float32)
    u1_jnp = jnp.array(np.array(u1, copy=True), dtype=jnp.complex64)
    u3_jnp = jnp.array(np.array(u3, copy=True), dtype=jnp.complex64)
    KZ_jnp = jnp.array(np.array(KZ, copy=True), dtype=jnp.float32)

    # Call the core - ensure NO static_argnums are accidentally set
    # print("d: ", d)
    # print("rho, ", rho)
    Ra_jax, Rb_jax, Rc_jax, Rd_jax = jax_magnetic_amplitude_core(
        d_jnp, sigma_jnp, rho_jnp, irho_jnp, rhoM_jnp, u1_jnp, u3_jnp, KZ_jnp
    )
    # DEBUG: Check if the JAX output itself is actually changing
    print(f"DEBUG: JAX Ra[20] = {Ra_jax[20]}, {rho_jnp[1]}")

    np.copyto(Ra, np.array(Ra_jax))
    np.copyto(Rb, np.array(Rb_jax))
    np.copyto(Rc, np.array(Rc_jax))
    np.copyto(Rd, np.array(Rd_jax))
