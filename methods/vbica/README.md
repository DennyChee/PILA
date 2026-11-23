# vbICA (adapter — Phase 2)

vbICA stays in its own repo (github.com/DennyChee/vbICA; local path = `${VBICA_REPO}`).

Adapter steps (decided 2026-11-23):
1. export the scenario to the vbICA `.mat` layout (legacy: `vbICA/Code_denny/pila_bench/convert_pila_cube_to_vbica.py`);
2. read `decomposedICA*.mat` back;
3. GROUP the ICs that belong to one deforming source (a moving source spreads over several ICs) with an
   honest, truth-free rule chosen in Phase 4; noise/APS ICs are left out by the same rule;
4. rebuild each source's per-date signal (sum of its ICs: map x time course);
5. invert every date of that signal with the same per-epoch SA+MCMC as `methods/mcmc`;
6. write the common `params.npz`.
