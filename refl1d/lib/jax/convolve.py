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
