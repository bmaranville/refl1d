import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def refl_jax(depth, sigma, rho, irho, kz_val):
    J = 1j
    PI4 = 12.566370614359172e-6
    cutoff = 1e-10
    L = len(depth)

    # 1. Determine direction based on kz
    is_forward = kz_val >= 0.0

    # 2. Pick the incident medium (Layer 0 if forward, Layer L-1 if backward)
    idx_incident = jnp.where(is_forward, 0, L - 1)

    k_init = jnp.complex64(jnp.abs(kz_val))
    kz_sq = jnp.complex64(kz_val**2 + PI4 * rho[idx_incident])

    init_state = (jnp.complex64(1.0), jnp.complex64(0.0), jnp.complex64(0.0), jnp.complex64(1.0), k_init)

    layer_indices = jnp.arange(L - 1)

    def scan_body(carry, step_idx):
        B11, B21, B12, B22, k_curr = carry

        # --- Array Reversal Logic ---
        # Map the step_idx (0 to L-2) to the correct array indices
        idx_rho = jnp.where(is_forward, step_idx + 1, L - 2 - step_idx)
        idx_sigma = jnp.where(is_forward, step_idx, L - 2 - step_idx)
        idx_depth = jnp.where(is_forward, step_idx, L - 1 - step_idx)

        arg = kz_sq - PI4 * (rho[idx_rho] + J * irho[idx_rho])

        # Critical edge branch cut fix
        is_evanescent = (irho[idx_rho] == 0.0) & (jnp.real(arg) < 0.0)
        k_next = jnp.where(is_evanescent, J * jnp.sqrt(jnp.abs(jnp.real(arg))), jnp.sqrt(arg))

        sig = sigma[idx_sigma]
        F = (k_curr - k_next) / (k_curr + k_next) * jnp.exp(-2.0 * k_curr * k_next * sig**2)

        d = depth[idx_depth]
        phase = jnp.where(step_idx > 0, jnp.exp(J * k_curr * d), jnp.complex64(1.0))
        inv_phase = jnp.where(step_idx > 0, jnp.exp(-J * k_curr * d), jnp.complex64(1.0))

        M11, M22 = phase, inv_phase
        M21, M12 = F * M11, F * M22

        new_B11 = B11 * M11 + B21 * M12
        new_B21 = B11 * M21 + B21 * M22
        new_B12 = B12 * M11 + B22 * M12
        new_B22 = B12 * M21 + B22 * M22

        return (new_B11, new_B21, new_B12, new_B22, k_next), None

    final_state, _ = lax.scan(scan_body, init_state, layer_indices)

    # Calculate R
    calc_r = final_state[2] / final_state[0]  # B12 / B11

    # 3. Handle the cutoff deadzone
    # If kz is between -cutoff and +cutoff, return -1 as in the original code
    return jnp.where(jnp.abs(kz_val) < cutoff, jnp.complex64(-1.0), calc_r)


def reflectivity_amplitude_jax(depth, sigma, rho_pool, irho_pool, kz, rho_index, r):
    """Refl1D backend signature compatibility wrapper."""
    rho_full = rho_pool[rho_index]
    irho_full = irho_pool[rho_index]

    res = jax.vmap(refl_jax, in_axes=(None, None, 0, 0, 0))(depth, sigma, rho_full, irho_full, kz)

    r[:] = res
    return r
