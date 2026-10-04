# Wafer Overlay Modeling, Robust Correction & Sampling Optimization

A Python study of **lithography overlay metrology**: how overlay errors are modeled, how correction terms are extracted from noisy measurements with outliers, and how the sampling plan changes the confidence in those corrections.

![Overlay vector maps](results/vector_maps.png)

---

## Why this matters

Every chip layer must line up with the one below it. **Overlay** is the misalignment between layers, measured as (dx, dy) at targets in the scribe lines. In advanced DRAM the overlay budget is only a few nanometres.

Overlay metrology tools measure these targets. The data is fitted to a model, and the fitted terms are fed back to the lithography scanner to correct later lots (advanced process control, APC). This project builds that loop from scratch and checks it against known ground truth.

---

## What the project does

| Step | What | Where |
|---|---|---|
| 1 | Build a 300 mm wafer exposure map (26 × 33 mm fields, 3 mm edge exclusion) with 4 overlay targets per field | `make_layout()` |
| 2 | Inject known scanner errors (translation, wafer/field magnification and rotation), 1 nm measurement noise and damaged-target outliers | `simulate_overlay()` |
| 3 | Fit the 10-parameter linear overlay model by least squares and report each term's standard error | `fit_plain()`, `param_uncertainty()` |
| 4 | Reject outliers with iterative 3σ filtering using a robust (MAD-based) sigma | `fit_robust()` |
| 5 | Compare sampling plans of equal cost by simulating 300 wafers per plan | `sampling_study.py` |

### The model

```
dx = Tx + Mx·X − Rx·Y + mx·x − rx·y
dy = Ty + My·Y + Ry·X + my·y + ry·x
```

X, Y are the field position on the wafer and x, y are the target position inside the field (mm). T is in nm; M, m are in ppm and R, r in µrad (1 ppm = 1 nm/mm).

---

## Key results

### 1. Robust correction recovers the true scanner errors

With 5 damaged targets (30–60 nm errors) among 228:

| Term | True | Plain fit | Robust fit | ±1σ |
|---|---|---|---|---|
| Tx (nm) | 3.000 | 2.455 | **2.959** | 0.059 |
| Ty (nm) | −2.000 | −2.719 | **−2.018** | 0.067 |
| My (ppm) | 0.0400 | 0.0316 | **0.0392** | 0.0012 |
| ry (µrad) | 0.1200 | 0.1003 | **0.1175** | 0.0061 |

- All 5 outliers were caught, with 1 false rejection out of 223 good targets.
- Overlay (|mean| + 3σ) went from **16.9 nm to 2.6 nm** in x and **13.3 nm to 3.0 nm** in y. What remains is the measurement-noise floor.

Full table: [`results/fit_summary.txt`](results/fit_summary.txt)

### 2. Where you measure matters as much as how much you measure

![Sampling study](results/sampling_study.png)

Repeatability of the wafer magnification term across 300 simulated wafers, with every reduced plan using 60 points:

| Plan | Mx 1σ (ppb) |
|---|---|
| All fields (228 pts) | 1.04 |
| Every 4th field | 1.83 |
| **15 outermost fields** | **1.35** |
| 15 innermost fields | 3.37 |

- Putting the same 60 points on **edge fields** gives about **26% lower** wafer-term uncertainty than uniform sampling, and **2.5× lower** than centre-heavy sampling.
- **Translation** depends only on the number of points, not their position.
- **Field terms** depend on how the targets are spread **inside each field**. Two targets on a diagonal make field magnification and rotation impossible to separate (the matrix becomes singular).

---

## Engineering takeaways

1. **Least squares is sensitive to outliers.** One 40 nm flyer among 7 points nearly tripled the fitted magnification. Robust rejection is essential before corrections go to the scanner.
2. **Threshold trade-off.** 2.5σ rejected about 5 good targets; 4σ rejected none but risks keeping medium outliers. 3σ is a sensible default.
3. **Outliers smaller than about 3× the noise can't be separated from noise**, but they barely affect the fit.
4. **Clustered outliers are information, not noise.** Three rejected targets in one field are unlikely by chance (about 1 in 300) and point to a local issue such as edge effects, chuck contamination, or target damage.
5. **Sampling design:** place wafer-level points far from the centre, spread targets in 2D within each field, and keep a few centre fields to catch non-linear signatures the linear model can't see.

---

## How to run

```bash
pip install -r requirements.txt
python run_demo.py        # simulation, plain vs robust fit, vector maps
python sampling_study.py  # sampling-plan comparison
```

Plots and the summary table are saved in `results/`.

## Files

```
overlay_model.py     core functions (layout, simulation, fitting, plotting)
run_demo.py          end-to-end correction demo
sampling_study.py    sampling-plan comparison
results/             generated figures and fit summary
```

## Possible extensions

- Higher-order (non-linear) wafer and field models, for example k-parameters up to 3rd order
- Per-exposure correction (CPE)
- Lot-to-lot APC feedback simulation with an EWMA controller

---

*Author: Vasudevan M C · [LinkedIn](https://linkedin.com/in/vasudevanmc)*
