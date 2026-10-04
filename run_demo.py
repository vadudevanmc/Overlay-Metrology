"""
run_demo.py
===========
End-to-end overlay correction demo:
    simulate a wafer -> plain fit -> robust fit -> compare -> save plots.

Run:  python run_demo.py
Outputs go to the results/ folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import overlay_model as om

# ----------------------------- settings --------------------------------------
SEED, NOISE = 42, 1.0             # random seed, measurement noise (nm, 1 sigma)
N_OUT, OUT_RANGE = 5, (30, 60)    # damaged targets and their size (nm)
K_SIGMA = 3.0                     # outlier rejection limit

TRUE = {"Tx": 3.0, "Ty": -2.0,    # translation (nm)
        "Mx": 0.05, "My": 0.04,   # wafer magnification (ppm)
        "Rx": 0.03, "Ry": 0.02,   # wafer rotation (urad)
        "mx": 0.20, "my": -0.15,  # field magnification (ppm)
        "rx": 0.10, "ry": 0.12}   # field rotation (urad)
# -----------------------------------------------------------------------------

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(SEED)

# 1. Layout and simulated measurement
centres, data = om.make_layout()
dx, dy, bad = om.simulate_overlay(data, TRUE, NOISE, N_OUT, OUT_RANGE, rng)
A_x, A_y = om.design_matrices(data)
print(f"Fields: {len(centres)}  Targets: {len(data)}  Damaged: {sorted(bad.tolist())}")

# 2. Plain and robust fits
px_plain, py_plain = om.fit_plain(A_x, dx), om.fit_plain(A_y, dy)
px_rob, keep_x = om.fit_robust(A_x, dx, K_SIGMA)
py_rob, keep_y = om.fit_robust(A_y, dy, K_SIGMA)
keep = keep_x & keep_y

se_x = om.param_uncertainty(A_x[keep_x], dx[keep_x], px_rob)
se_y = om.param_uncertainty(A_y[keep_y], dy[keep_y], py_rob)

# 3. Outlier scorecard
rejected = set(np.where(~keep)[0].tolist())
caught = rejected & set(bad.tolist())
print(f"Outliers caught: {len(caught)}/{N_OUT}   "
      f"false alarms: {sorted(rejected - set(bad.tolist()))}")

# 4. Parameter table
plain = dict(zip(om.NAMES_X, px_plain)) | dict(zip(om.NAMES_Y, py_plain))
robust = dict(zip(om.NAMES_X, px_rob)) | dict(zip(om.NAMES_Y, py_rob))
se = dict(zip(om.NAMES_X, se_x)) | dict(zip(om.NAMES_Y, se_y))

lines = [f"{'term':<5}{'true':>9}{'plain':>10}{'robust':>10}{'±1σ':>9}"]
for k in TRUE:
    lines.append(f"{k:<5}{TRUE[k]:>9.4f}{plain[k]:>10.4f}{robust[k]:>10.4f}{se[k]:>9.4f}")

# 5. Overlay before / after correction
res_x, res_y = dx - A_x @ px_rob, dy - A_y @ py_rob
lines += ["",
          f"Raw overlay      |mean|+3σ: x = {om.overlay_metric(dx[keep]):.2f} nm, "
          f"y = {om.overlay_metric(dy[keep]):.2f} nm",
          f"Residual overlay |mean|+3σ: x = {om.overlay_metric(res_x[keep]):.2f} nm, "
          f"y = {om.overlay_metric(res_y[keep]):.2f} nm"]
report = "\n".join(lines)
print("\n" + report)
with open("results/fit_summary.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n")

# 6. Plots
om.plot_vector_maps(
    data, centres,
    [(dx, dy, "Raw (measured)"),
     (A_x @ px_rob, A_y @ py_rob, "Model (robust fit)"),
     (res_x, res_y, "Residual after correction")],
    keep=keep,
    title="Overlay vector maps (red = rejected outlier targets)",
    savepath="results/vector_maps.png")

plt.show()
