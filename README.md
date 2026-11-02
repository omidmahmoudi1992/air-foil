# NACA 0012 airfoil: RANS lift curve and four-model setups (COMSOL + Python)

This is a small self-study project from October 2026. The goal was to learn how much the
answer for one airfoil changes when the turbulence treatment changes: steady RANS,
unsteady RANS (URANS), DES and LES. A public post with a four-panel fidelity comparison
for this airfoil gave me the idea. I built the cases in COMSOL Multiphysics 6.3 and
control the runs with Python (MPh library).

## Case conditions

- Airfoil: NACA 0012, chord c = 1 m, kept horizontal; the angle of attack is set by the
  inlet velocity direction (parameters `u0`, `v0`).
- Reynolds number Re = 500,000; air with rho = 1.225 kg/m^3 and mu = 1.787e-5 Pa*s,
  so U0 is about 7.30 m/s.
- Turbulence model in the 2D base model: Turbulent Flow, k-omega SST (fully turbulent,
  no transition model).
- Lift comes from a global variable probe on the airfoil surface:
  `-int_air(spf.T_stressy*cos(aoa) - spf.T_stressx*sin(aoa))`, i.e. the surface stress
  force turned into wind axes. Cl = Lift / (q * c) with q = 0.5 * rho * U0^2.

## What is finished, and what is not

| Part | Status |
|---|---|
| 2D steady RANS lift curve, alpha = 0 to 25 deg | **Done** (results below) |
| Angle-of-attack sweep script with plots | **Done** (`comsol_aoa_sweep.py`) |
| Time-budget mesh-size control loop | **Done as a tool** (`comsol_adaptive_loop.py`); it is not a mesh-independence proof |
| URANS 2D | Setup file exists; **no results yet** |
| DES / LES 3D pilot setups (periodic span 0.2c) | Setup files built by script; **not meshed and not solved** |
| Drag coefficient | **Not yet** (the drag probe was missing in the recorded sweep, so Cd is NaN in the CSV) |
| Formal mesh-independence study | **Not yet** (mesh-check runs at 10 and 15 deg exist locally; summary still to be added) |
| Comparison with published experimental data | **Not yet** |

My personal computer needs about 400 to 900 seconds for one steady 2D angle, so the 3D
cases must wait for stronger hardware. The 3D files are templates, not validated
simulations.

## Results so far (steady RANS)

From `results/aoa_sweep_results.csv` (plot: `results/aoa_sweep_Cl_alpha.png`):

| alpha [deg] | Cl |
|---|---|
| 0 | -0.000 |
| 5 | 0.459 |
| 10 | 0.879 |
| 15 | 1.138 |
| 20 | 0.607 |
| 25 | 0.606 |

Notes on these numbers:

- The lift slope in the linear part is about 0.09 per degree and the curve turns down
  after 15 degrees. This matches the known behaviour of this airfoil in a general way.
- The values at 20 and 25 degrees come from a **steady** solver in deep stall. A steady
  solution there is not physical; URANS is the minimum for those angles. I keep the
  numbers in the table because the CSV is the raw record of the run, but I do not use
  them for conclusions.
- The model is fully turbulent. At Re = 500,000 the real flow has laminar-turbulent
  transition, so Cl,max from this setup can sit above wind-tunnel values at the same
  Reynolds number. A direct comparison with a published data set is still on my list.

## Files

- `comsol_aoa_sweep.py` — sets each angle of attack, solves the stationary study with
  continuation from the last solution, reads the lift (and drag, if the probe exists),
  writes the CSV after every angle and draws the plots. Optional URANS step with
  time-averaging of the lift tail.
- `comsol_adaptive_loop.py` — controls the mesh size parameter `hmax`: solve, then
  refine when the solve is fast, coarsen when it passes a time budget, stop when the
  lift change is under 2 percent. `results/adaptive_log.csv` is one recorded run at
  25 degrees; it stopped at the coarse cap hmax = 0.150 m, so it does **not** prove
  mesh independence.
- `comsol_build_four_models.py` — creates the four setup files from my 2D base model:
  RANS 2D (copy), URANS 2D (adds a transient study), and two extruded 3D pilot setups
  with periodic span faces (DES RBVM Spalart-Allmaras and LES RBVM interfaces). The
  script never meshes and never solves; its docstring says this in plain words.
- `results/` — the recorded sweep CSV, the adaptive-loop log and the lift-curve plot.

## Requirements

- COMSOL Multiphysics 6.3 with the CFD Module and a valid licence.
- Python with `mph`, `numpy` and `matplotlib` (the build script also uses `jpype`).
- My 2D base model file (`.mph`) is not in this repository. The scripts expect a model
  with global parameters `c`, `Re`, `rho`, `mu`, `aoa`, `U0`, `u0`, `v0`, a lift probe
  named `Lift`, and a stationary study named `Study 1`. The docstring at the top of
  each script lists the full setup.

## Next steps

1. Add the drag probe and repeat the sweep, so the drag polar exists.
2. Run one finer mesh at 10 and 15 degrees and report the Cl change honestly.
3. Run URANS at 20 and 25 degrees and compare the mean lift with the steady values.
4. Compare the lift curve with a published experimental data set at Re = 500,000.
5. Solve the 3D DES/LES pilots when I have access to stronger computing resources.

Small note: in these copies, one local Windows file path in
`comsol_build_four_models.py` was changed to a relative path. Nothing else was changed.
