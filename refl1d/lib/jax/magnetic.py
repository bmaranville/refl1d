import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def jax_Cr4xa_single(d, sigma, IP, rho, irho, rhoM, u1, u3, kz):
    """
    JAX implementation of the Cr4xa Parratt recursion for a single KZ point.
    """
    EPS = 1e-10
    PI4 = 4.0e-6 * jnp.pi

    # 1. Handle KZ directionality cleanly by conditionally flipping arrays
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

    # Initialize variables for L=0
    S1L_init = get_S(rho_use[0], rhoM_use[0], irho_use[0])
    S3L_init = get_S3(rho_use[0], rhoM_use[0], irho_use[0])

    flip_cond_init = jnp.abs(u1_use[0]) > 1.0
    S1L_init, S3L_init = jnp.where(flip_cond_init, S3L_init, S1L_init), jnp.where(flip_cond_init, S1L_init, S3L_init)

    BL_init = 0.0 + 0j
    GL_init = 0.0 + 0j
    Z_init = 0.0
    B_init = jnp.eye(4, dtype=jnp.complex128)

    # 2. General Abeles Matrix recursion
    def scan_step(carry, i):
        B_mat, S1L, S3L, BL, GL, Z = carry
        LP = i + 1

        S1LP = get_S(rho_use[LP], rhoM_use[LP], irho_use[LP])
        S3LP = get_S3(rho_use[LP], rhoM_use[LP], irho_use[LP])

        flip_cond = jnp.abs(u1_use[LP]) > 1.0
        S1LP, S3LP = jnp.where(flip_cond, S3LP, S1LP), jnp.where(flip_cond, S1LP, S3LP)

        BLP = jnp.where(flip_cond, u3_use[LP], u1_use[LP])
        GLP = jnp.where(flip_cond, 1.0 / (u1_use[LP] + EPS), 1.0 / (u3_use[LP] + EPS))

        SIGMAL = sigma_use[i]
        DELTA = 0.5 / (1.0 - (BLP * GLP))

        DBB = (BL - BLP) * DELTA
        DBG = (1.0 - BL * GLP) * DELTA
        DGB = (1.0 - GL * BLP) * DELTA
        DGG = (GL - GLP) * DELTA

        ES1L = jnp.exp(S1L * Z)
        ENS1L = 1.0 / ES1L
        ES1LP = jnp.exp(S1LP * Z)
        ENS1LP = 1.0 / ES1LP
        ES3L = jnp.exp(S3L * Z)
        ENS3L = 1.0 / ES3L
        ES3LP = jnp.exp(S3LP * Z)
        ENS3LP = 1.0 / ES3LP

        FS1S1 = S1L / S1LP
        FS1S3 = S1L / S3LP
        FS3S1 = S3L / S1LP
        FS3S3 = S3L / S3LP

        # Populate Abeles matrix components
        A11 = DBG * (1.0 + FS1S1) * ES1L * ENS1LP
        A22 = DBG * (1.0 + FS1S1) * ENS1L * ES1LP
        A12 = DBG * (1.0 - FS1S1) * jnp.exp(2.0 * S1L * S1LP * SIGMAL**2) * ENS1L * ENS1LP
        A21 = DBG * (1.0 - FS1S1) * jnp.exp(2.0 * S1L * S1LP * SIGMAL**2) * ES1L * ES1LP

        A13 = DGG * (1.0 + FS3S1) * ES3L * ENS1LP
        A24 = DGG * (1.0 + FS3S1) * ENS3L * ES1LP
        A14 = DGG * (1.0 - FS3S1) * jnp.exp(2.0 * S3L * S1LP * SIGMAL**2) * ENS3L * ENS1LP
        A23 = DGG * (1.0 - FS3S1) * jnp.exp(2.0 * S3L * S1LP * SIGMAL**2) * ES3L * ES1LP

        A31 = DBB * (1.0 + FS1S3) * ES1L * ENS3LP
        A42 = DBB * (1.0 + FS1S3) * ENS1L * ES3LP
        A32 = DBB * (1.0 - FS1S3) * jnp.exp(2.0 * S1L * S3LP * SIGMAL**2) * ENS1L * ENS3LP
        A41 = DBB * (1.0 - FS1S3) * jnp.exp(2.0 * S1L * S3LP * SIGMAL**2) * ES1L * ES3LP

        A33 = DGB * (1.0 + FS3S3) * ES3L * ENS3LP
        A44 = DGB * (1.0 + FS3S3) * ENS3L * ES3LP
        A34 = DGB * (1.0 - FS3S3) * jnp.exp(2.0 * S3L * S3LP * SIGMAL**2) * ENS3L * ENS3LP
        A43 = DGB * (1.0 - FS3S3) * jnp.exp(2.0 * S3L * S3LP * SIGMAL**2) * ES3L * ES3LP

        A = jnp.array([[A11, A12, A13, A14], [A21, A22, A23, A24], [A31, A32, A33, A34], [A41, A42, A43, A44]])

        new_B_mat = A @ B_mat
        new_Z = Z + d_use[LP]

        return (new_B_mat, S1LP, S3LP, BLP, GLP, new_Z), None

    # 3. Scan the matrix multiplications across all interfaces (N - 1 times)
    N = d.shape[0]
    init_carry = (B_init, S1L_init, S3L_init, BL_init, GL_init, Z_init)
    final_carry, _ = jax.lax.scan(scan_step, init_carry, jnp.arange(N - 1))
    B_final = final_carry[0]

    # 4. Compute reflectivities
    DETW = B_final[3, 3] * B_final[1, 1] - B_final[1, 3] * B_final[3, 1]

    YA = jnp.where(IP > 0, (B_final[1, 3] * B_final[3, 0] - B_final[1, 0] * B_final[3, 3]) / DETW, 0.0j)
    YB = jnp.where(IP > 0, (B_final[1, 0] * B_final[3, 1] - B_final[3, 0] * B_final[1, 1]) / DETW, 0.0j)
    YC = (B_final[1, 3] * B_final[3, 2] - B_final[1, 2] * B_final[3, 3]) / DETW
    YD = (B_final[1, 2] * B_final[3, 1] - B_final[3, 2] * B_final[1, 1]) / DETW

    # Handle the empty/zero KZ case boundary
    is_valid = jnp.abs(kz) >= 1e-10
    YA = jnp.where(is_valid, YA, -1.0 + 0j)
    YB = jnp.where(is_valid, YB, 0.0 + 0j)
    YC = jnp.where(is_valid, YC, 0.0 + 0j)
    YD = jnp.where(is_valid, YD, -1.0 + 0j)

    return YA, YB, YC, YD


@jax.jit
def jax_magnetic_amplitude_core(d, sigma, rho, irho, rhoM, u1, u3, KZ):
    """
    Vectorizes Cr4xa over the KZ array for both IP = 1 and IP = -1.
    """
    vmap_fn = jax.vmap(jax_Cr4xa_single, in_axes=(None, None, None, None, None, None, None, None, 0))

    # Calculate both polarizations
    Ra_p, Rb_p, Rc_p, Rd_p = vmap_fn(d, sigma, 1.0, rho, irho, rhoM, u1, u3, KZ)
    Ra_m, Rb_m, Rc_m, Rd_m = vmap_fn(d, sigma, -1.0, rho, irho, rhoM, u1, u3, KZ)

    # In the original python code, the IP = -1 loop only overwrites Rc and Rd
    # So we combine Ra and Rb from IP = 1 with Rc and Rd from IP = -1
    return Ra_p, Rb_p, Rc_m, Rd_m


def magnetic_amplitude(d, sigma, rho, irho, rhoM, u1, u3, KZ, rho_index, Ra, Rb, Rc, Rd):
    """
    JAX-accelerated drop-in replacement for refl1d's magnetic_amplitude.
    Calculates R++, R+-, R-+, R-- simultaneously and updates the output arrays in-place.
    """
    # 1. Cast everything explicitly to 64-bit structures
    d_jnp = jnp.asarray(d, dtype=jnp.float64)
    sigma_jnp = jnp.asarray(sigma, dtype=jnp.float64)
    rho_jnp = jnp.asarray(rho, dtype=jnp.float64)
    irho_jnp = jnp.asarray(irho, dtype=jnp.float64)
    rhoM_jnp = jnp.asarray(rhoM, dtype=jnp.float64)

    # 64-bit complex is complex128 (two float64s)
    u1_jnp = jnp.asarray(u1, dtype=jnp.complex128)
    u3_jnp = jnp.asarray(u3, dtype=jnp.complex128)
    KZ_jnp = jnp.asarray(KZ, dtype=jnp.float64)

    # 2. Run the JIT-compiled computation on the MLX/MPS backend
    Ra_jax, Rb_jax, Rc_jax, Rd_jax = jax_magnetic_amplitude_core(
        d_jnp, sigma_jnp, rho_jnp, irho_jnp, rhoM_jnp, u1_jnp, u3_jnp, KZ_jnp
    )

    # 3. Explicitly copy the results back into the provided memory addresses
    np.copyto(Ra, np.asarray(Ra_jax))
    np.copyto(Rb, np.asarray(Rb_jax))
    np.copyto(Rc, np.asarray(Rc_jax))
    np.copyto(Rd, np.asarray(Rd_jax))
