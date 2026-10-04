"""
sampling_study.py
=================
Compare overlay sampling plans with equal measurement cost.

For each plan, simulate many wafers and measure how much each fitted
parameter scatters (its repeatability). Lower scatter = better plan.

Run:  python sampling_study.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import overlay_model as om

N_WAFERS = 300
NOISE = 1.0
TRUE = {"Tx": 3.0, "Ty": -2.0, "Mx": 0.05, "My": 0.04, "Rx": 0.03, "Ry": 0.02,
        "mx": 0.20, "my": -0.15, "rx": 0.10, "ry": 0.12}

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(7)

centres, full = om.make_layout()
radius = np.hypot(centres[:, 0], centres[:, 1])
order = np.argsort(radius)                      # field ids, nearest -> farthest

plans = {
    "All fields (228 pts)":       np.arange(len(centres)),
    "Every 4th field (60 pts)":   np.arange(len(centres))[::4],
    "15 outermost (60 pts)":      order[-15:],
    "14 outer + centre (60 pts)": np.r_[order[-14:], order[0]],
    "15 innermost (60 pts)":      order[:15],
}

results = {}
for name, fields in plans.items():
    data = full[np.isin(full[:, 0], fields)]
    A_x, A_y = om.design_matrices(data)
    fits = []
    for _ in range(N_WAFERS):
        dx, dy, _ = om.simulate_overlay(data, TRUE, NOISE, rng=rng)
        fits.append(np.r_[om.fit_plain(A_x, dx), om.fit_plain(A_y, dy)])
    results[name] = np.array(fits).std(axis=0)   # scatter of each parameter

terms = om.NAMES_X + om.NAMES_Y
show = ["Tx", "Mx", "My", "Rx", "Ry", "mx"]
idx = [terms.index(t) for t in show]

print(f"Parameter scatter (1σ) over {N_WAFERS} simulated wafers\n")
print(f"{'plan':<28}" + "".join(f"{t:>9}" for t in show))
for name, s in results.items():
    print(f"{name:<28}" + "".join(f"{s[i]:>9.4f}" for i in idx))

# Bar chart: wafer magnification Mx scatter per plan
fig, ax = plt.subplots(figsize=(8, 4))
names = list(results)
vals = [results[n][terms.index("Mx")] * 1000 for n in names]
ax.barh(names, vals)
ax.invert_yaxis()
ax.set_xlabel("Mx repeatability, 1σ (ppb)")
ax.set_title("Wafer magnification uncertainty by sampling plan")
for i, v in enumerate(vals):
    ax.text(v, i, f" {v:.2f}", va="center")
fig.tight_layout()
fig.savefig("results/sampling_study.png", dpi=150)
plt.show()
