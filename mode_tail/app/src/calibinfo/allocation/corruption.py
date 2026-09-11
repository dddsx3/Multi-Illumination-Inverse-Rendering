"""Paired corruption draws with per-light precision scaling.

Functional forms are exactly those of calibinfo.models.corruption (rotate_dirs,
scale_intensity); the difference is that the raw standard-normal innovations can be
drawn once and scaled per light, so that every policy/budget sees the SAME raw
innovations (paired comparison, frozen seed construction).

With scales == 1, `paired_corruption` is bit-identical to
CorruptionGenerator.apply on the same rng state (binding-tested), and
`apply_scaled_corruption` with pre-drawn innovations equals the same realization.
"""

from __future__ import annotations

import numpy as np


def raw_innovations(rng, n):
    """Shared raw standard-normal innovations for one corruption realization.

    Draw order mirrors calibinfo.models.corruption exactly: per light an axis
    vector (3 draws) then a raw angle, interleaved, then the n raw log-intensity
    innovations as one vector.
    """
    axes = np.empty((n, 3))
    angles = np.empty(n)
    for i in range(n):
        axes[i] = rng.normal(size=3)
        angles[i] = rng.normal(size=1)[0]
    logs = rng.normal(size=n)
    return axes, angles, logs


def apply_scaled_corruption(dirs, sig_logI, sig_rad, scales, raw):
    """Apply the corruption functional form with per-light std scales.

    dirs: (n, 3); scales: (n,) multiplicative factors on the per-light standard
    deviation (1 = no allocation, 1/sqrt(regime) = selected); raw = the shared
    (axes, angles, logs) innovations. Returns (dirs_corrupted (n, 3), gains (n,)).
    """
    dirs = np.atleast_2d(np.asarray(dirs, float))
    n = dirs.shape[0]
    scales = np.broadcast_to(np.asarray(scales, float), (n,))
    axes, angles, logs = raw
    if sig_rad > 0:
        d2 = np.empty_like(dirs)
        for i in range(n):
            v = axes[i].copy()
            v -= dirs[i] * (v @ dirs[i])
            nv = np.linalg.norm(v)
            v = v / nv if nv > 1e-12 else np.array([1.0, 0.0, 0.0])
            ang = angles[i] * sig_rad * scales[i]
            out = dirs[i] * np.cos(ang) + np.cross(v, dirs[i]) * np.sin(ang) \
                + dirs[i] * (dirs[i] @ v) * (1 - np.cos(ang)) * 0.0
            d2[i] = out / np.linalg.norm(out)
    else:
        d2 = dirs.copy()
    if sig_logI > 0:
        g = np.exp(logs * sig_logI * scales)
    else:
        g = np.ones(n)
    return d2, g


def paired_corruption(rng, dirs, sig_logI, sig_rad, scales):
    """Convenience wrapper: draw the shared innovations and apply them."""
    dirs = np.atleast_2d(np.asarray(dirs, float))
    raw = raw_innovations(rng, dirs.shape[0])
    return apply_scaled_corruption(dirs, sig_logI, sig_rad, scales, raw)
