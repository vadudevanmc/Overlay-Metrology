"""
overlay_model.py
================
Core functions for wafer overlay modeling.

    1. make_layout()      - scanner exposure map + overlay targets
    2. simulate_overlay() - inject known scanner errors, noise and outliers
    3. design_matrices()  - build the 10-parameter linear model tables
    4. fit_plain()        - ordinary least squares
    5. fit_robust()       - iterative k-sigma outlier rejection (MAD sigma)
    6. param_uncertainty()- standard error of every fitted term
    7. overlay_metric()   - |mean| + 3 sigma
    8. plot_vector_maps() - raw / model / residual vector maps

Model (units: X, Y, x, y in mm; dx, dy in nm; coefficients in nm/mm = ppm or urad)

    dx = Tx + Mx*X - Rx*Y + mx*x - rx*y
    dy = Ty + My*Y + Ry*X + my*y + ry*x

Author: Vasudevan M C
"""

import numpy as np
import matplotlib.pyplot as plt

NAMES_X = ["Tx", "Mx", "Rx", "mx", "rx"]
NAMES_Y = ["Ty", "My", "Ry", "my", "ry"]


# -----------------------------------------------------------------------------
# 1. Layout
# -----------------------------------------------------------------------------
def make_layout(wafer_r=150.0, edge_excl=3.0, field_w=26.0, field_h=33.0,
                local_targets=((-11, -14), (11, -14), (-11, 14), (11, 14))):
    """Return (centres, data).

    centres : (n_fields, 2) array of field centres (X, Y)
    data    : (n_targets, 7) array with columns
              [field_id, X, Y, x, y, abs_X, abs_Y]
    """
    centres = []
    for i in range(-6, 7):
        for j in range(-5, 6):
            cx, cy = i * field_w, j * field_h
            corners = [(cx + sx * field_w / 2, cy + sy * field_h / 2)
                       for sx in (-1, 1) for sy in (-1, 1)]
            if all(np.hypot(px, py) <= wafer_r - edge_excl for px, py in corners):
                centres.append((cx, cy))
    centres = np.array(centres)

    local = np.array(local_targets, dtype=float)
    rows = [[f, cx, cy, lx, ly, cx + lx, cy + ly]
            for f, (cx, cy) in enumerate(centres)
            for lx, ly in local]
    return centres, np.array(rows)


# -----------------------------------------------------------------------------
# 2. Simulation
# -----------------------------------------------------------------------------
def simulate_overlay(data, true, noise=1.0, n_out=0, out_range=(30, 60), rng=None):
    """Generate measured dx, dy from 'true' parameters + noise + outliers.

    Returns dx, dy and the indices of the damaged (outlier) targets.
    """
    rng = rng if rng is not None else np.random.default_rng()
    X, Y, x, y = data[:, 1], data[:, 2], data[:, 3], data[:, 4]

    dx = true["Tx"] + true["Mx"] * X - true["Rx"] * Y + true["mx"] * x - true["rx"] * y
    dy = true["Ty"] + true["My"] * Y + true["Ry"] * X + true["my"] * y + true["ry"] * x
    dx = dx + rng.normal(0, noise, len(dx))
    dy = dy + rng.normal(0, noise, len(dy))

    bad = rng.choice(len(dx), size=n_out, replace=False) if n_out > 0 else np.array([], int)
    if n_out > 0:
        lo, hi = out_range
        dx[bad] += rng.choice([-1, 1], n_out) * rng.uniform(lo, hi, n_out)
        dy[bad] += rng.choice([-1, 1], n_out) * rng.uniform(lo, hi, n_out)
    return dx, dy, bad


# -----------------------------------------------------------------------------
# 3. Design matrices
# -----------------------------------------------------------------------------
def design_matrices(data):
    """Return (A_x, A_y): one row per target, one column per parameter."""
    X, Y, x, y = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    ones = np.ones_like(X)
    A_x = np.column_stack([ones, X, -Y, x, -y])   # Tx, Mx, Rx, mx, rx
    A_y = np.column_stack([ones, Y,  X, y,  x])   # Ty, My, Ry, my, ry
    return A_x, A_y


# -----------------------------------------------------------------------------
# 4-5. Fitting
# -----------------------------------------------------------------------------
def fit_plain(A, d):
    """Ordinary least squares using every target."""
    return np.linalg.lstsq(A, d, rcond=None)[0]


def fit_robust(A, d, k=3.0, use_mad=True, max_iter=10):
    """Iterative k-sigma outlier rejection.

    Returns (params, keep) where keep is a boolean mask (True = target used).
    """
    keep = np.ones(len(d), dtype=bool)
    for _ in range(max_iter):
        p = np.linalg.lstsq(A[keep], d[keep], rcond=None)[0]
        res = d - A @ p
        r = res[keep]
        sigma = (1.4826 * np.median(np.abs(r - np.median(r)))) if use_mad else r.std()
        new_keep = np.abs(res) < k * sigma
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return p, keep


# -----------------------------------------------------------------------------
# 6-7. Statistics
# -----------------------------------------------------------------------------
def param_uncertainty(A, d, p):
    """Standard error (1 sigma) of each fitted parameter."""
    res = d - A @ p
    dof = len(d) - A.shape[1]
    sigma = np.sqrt(np.sum(res ** 2) / dof)
    return sigma * np.sqrt(np.diag(np.linalg.inv(A.T @ A)))


def overlay_metric(d):
    """Fab overlay number: |mean| + 3 sigma."""
    return abs(d.mean()) + 3 * d.std()


# -----------------------------------------------------------------------------
# 8. Plotting
# -----------------------------------------------------------------------------
def plot_vector_maps(data, centres, panels, keep=None, wafer_r=150.0,
                     field_w=26.0, field_h=33.0, nm_per_mm=1.5,
                     title="", savepath=None):
    """Draw side-by-side overlay vector maps.

    panels : list of (u, v, subtitle)
    keep   : boolean mask; rejected targets are drawn in red
    """
    keep = np.ones(len(data), bool) if keep is None else keep
    fig, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels), 5.5))
    axes = np.atleast_1d(axes)
    for ax, (u, v, sub) in zip(axes, panels):
        ax.add_patch(plt.Circle((0, 0), wafer_r, fill=False))
        for cx, cy in centres:
            ax.add_patch(plt.Rectangle((cx - field_w / 2, cy - field_h / 2),
                                       field_w, field_h, fill=False, lw=0.3, color="0.7"))
        q = ax.quiver(data[keep, 5], data[keep, 6], u[keep], v[keep],
                      angles="xy", scale_units="xy", scale=1 / nm_per_mm, width=0.004)
        if (~keep).any():
            ax.quiver(data[~keep, 5], data[~keep, 6],
                      np.clip(u[~keep], -15, 15), np.clip(v[~keep], -15, 15),
                      color="red", angles="xy", scale_units="xy",
                      scale=1 / nm_per_mm, width=0.005)
            ax.plot(data[~keep, 5], data[~keep, 6], "o", mfc="none", mec="red", ms=9)
        ax.quiverkey(q, 0.78, 1.03, 10, "10 nm", labelpos="E")
        ax.set_aspect("equal")
        ax.set_xlim(-wafer_r - 10, wafer_r + 10)
        ax.set_ylim(-wafer_r - 10, wafer_r + 10)
        ax.set_xlabel("X (mm)")
        ax.set_title(sub)
    axes[0].set_ylabel("Y (mm)")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
