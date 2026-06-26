#!/usr/bin/env python
# Usage:       python make_comparison_figure.py
# Description: Build a single presentation figure summarizing the PILA-vs-classical
#              (SA + Bayesian MCMC) comparison: accuracy parity, per-scene latency,
#              the amortization/scaling curve (how PILA scales up over a time series),
#              and the training-inclusive break-even / total-cost story.
# Date:        2026-06-22

import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless HPC node — no display
import matplotlib.pyplot as plt
from adjustText import adjust_text  # repels overlapping point labels with leader lines

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(CURRENT_DIR, "comparison_out", "pila_vs_classical_summary.png")
OUT_PDF = os.path.join(CURRENT_DIR, "comparison_out", "pila_vs_classical_summary.pdf")


def load_training_seconds(config_name):
    """Most-recent run's training wall-time (sec) + n stack epochs, from compute_usage.json."""
    hits = glob.glob(os.path.join(CURRENT_DIR, "saved", "*", config_name, "*", "models", "compute_usage.json"))
    if not hits:
        return np.nan, np.nan
    latest = sorted(hits, key=lambda p: p.split(os.sep)[-3])[-1]  # sort by run timestamp dir
    with open(latest) as fh:
        u = json.load(fh)
    return float(u.get("train_sec", np.nan)), int(u.get("n_samples_epochs", 0))


def main():
    # --- [1/4] Load the joined timing table + PILA accuracy metrics + classical R2 ---
    print("[1/4] Loading timing, accuracy and training data...")
    timing = pd.read_csv(os.path.join(CURRENT_DIR, "comparison_out", "classical_vs_pila_timing.csv"))
    bench = pd.read_csv(os.path.join(CURRENT_DIR, "comparison_out", "pila_inference_benchmark.csv"))
    comp = pd.read_csv(os.path.join(CURRENT_DIR, "comparison_out", "comparison_table.csv"))

    # PILA physics-only R2 (apples-to-apples vs classical) from the inference benchmark
    pila_r2 = bench.set_index("config")["r2_phys"]
    # Bayesian MCMC R2 (gold-standard classical) from the comparison table
    bayes_r2 = (comp[comp["method"] == "Bayesian"].set_index("config")["r2"]).astype(float)

    # Training wall-time + stack length per config
    train_s, n_ep = {}, {}
    for cfg in timing["config"]:
        ts, ne = load_training_seconds(cfg)
        train_s[cfg], n_ep[cfg] = ts, ne
    timing["train_s"] = timing["config"].map(train_s)
    timing["n_ep"] = timing["config"].map(n_ep)
    timing["pila_s"] = timing["pila_infer_ms_median"] / 1000.0
    # Break-even (scenes to recoup training) and full-stack total-cost speedup incl. training
    timing["breakeven"] = timing["train_s"] / (timing["runtime_total_sec"] - timing["pila_s"])
    timing["stack_speedup"] = (timing["n_ep"] * timing["runtime_total_sec"]) / (
        timing["train_s"] + timing["n_ep"] * timing["pila_s"]
    )

    # Color by source model for visual grouping
    def model_of(cfg):
        c = cfg.lower()
        return "Okada" if "okada" in c else ("Sun69" if "sun69" in c else "Mogi")

    timing["model"] = timing["config"].map(model_of)
    model_color = {"Mogi": "#1f77b4", "Sun69": "#2ca02c", "Okada": "#d62728"}

    med_marg = timing["speedup"].median()
    med_stack = timing["stack_speedup"].median()
    med_break = timing["breakeven"].median()
    med_train = timing["train_s"].median()
    med_clas = timing["runtime_total_sec"].median()
    med_pila_ms = timing["pila_infer_ms_median"].median()

    # --- [2/4] Build the 2x2 figure ---
    print("[2/4] Drawing panels...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axA, axB, axC, axD = axes.ravel()

    # Panel A: accuracy parity — PILA vs Bayesian MCMC physics-only R^2 (no accuracy cost)
    # short label per config: volcano abbrev + model + track, e.g. "Ny Okada TA174"
    volc_abbr = {"etna": "Et", "lapalma": "LP", "nyiragongo": "Ny"}

    def short_label(cfg):
        c = cfg.lower()
        v = next((a for k, a in volc_abbr.items() if k in c), cfg[:2])
        track = cfg.replace("_A", "").split("_")[-1]  # e.g. TA174
        return f"{v} {model_of(cfg)} {track}"

    xs, ys, cs, labels = [], [], [], []
    for cfg in timing["config"]:
        if cfg in pila_r2.index and cfg in bayes_r2.index:
            xs.append(bayes_r2[cfg]); ys.append(pila_r2[cfg])
            cs.append(model_color[model_of(cfg)]); labels.append(short_label(cfg))
    lim = [-0.5, 0.85]
    axA.plot(lim, lim, "k--", lw=1, label="1:1 (parity)")
    axA.scatter(xs, ys, c=cs, s=60, edgecolor="k", linewidth=0.4, zorder=3)
    axA.set_xlim(lim); axA.set_ylim(lim)
    # adjustText: repel labels so none overlap, drawing thin leader lines back to each dot
    texts = [axA.text(x, y, lab, fontsize=6, color="k", zorder=5)
             for x, y, lab in zip(xs, ys, labels)]
    adjust_text(texts, x=xs, y=ys, ax=axA,
                expand=(1.25, 1.4), force_text=(0.4, 0.5),
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4))
    axA.set_xlabel("Bayesian MCMC  R²  (physics-only)")
    axA.set_ylabel("PILA  R²  (physics-only)")
    axA.set_title("A. Accuracy parity — PILA ≈ gold-standard MCMC\n(points near 1:1 ⇒ no accuracy cost for the speedup)")
    axA.legend(loc="upper left", fontsize=9)
    axA.grid(alpha=0.3)

    # Panel B: per-scene wall-time (classical optimisation vs PILA inference), log scale
    order = timing.sort_values("runtime_total_sec").reset_index(drop=True)
    ypos = np.arange(len(order))
    axB.barh(ypos + 0.2, order["runtime_total_sec"], height=0.4, color="#888888", label="Classical SA+MCMC (per scene)")
    axB.barh(ypos - 0.2, order["pila_s"], height=0.4, color="#ff7f0e", label="PILA inference (per scene)")
    axB.set_yticks(ypos)
    axB.set_yticklabels(order["config"].str.replace("_A", "", regex=False), fontsize=7)
    axB.set_xscale("log")
    axB.set_xlabel("Wall-time per scene (s, log scale)")
    axB.set_title(f"B. Per-scene cost — median {med_clas:.0f} s vs {med_pila_ms:.2f} ms\n→ marginal speedup median {med_marg:,.0f}×")
    axB.legend(loc="lower right", fontsize=8)
    axB.grid(alpha=0.3, axis="x")

    # Panel C: SCALING / amortization — cumulative cost vs number of scenes (the "scale up" story)
    N = np.arange(0, 401)
    # representative configs: cheapest and most expensive classical, plus medians
    cheap = timing.loc[timing["runtime_total_sec"].idxmin()]
    exp = timing.loc[timing["runtime_total_sec"].idxmax()]
    for row, col, name in [(cheap, "#1f77b4", "cheap scene"), (exp, "#d62728", "expensive scene")]:
        clas_curve = N * row["runtime_total_sec"]
        pila_curve = row["train_s"] + N * row["pila_s"]
        axC.plot(N, clas_curve, color=col, ls="-", lw=2,
                 label=f"Classical — {row['config'].replace('_A','')} ({name})")
        axC.plot(N, pila_curve, color=col, ls="--", lw=2,
                 label=f"PILA (train+infer) — same config")
        be = row["breakeven"]
        axC.axvline(be, color=col, lw=0.8, alpha=0.5)
    axC.set_yscale("log")
    axC.set_xlabel("Number of scenes (time-series epochs) inverted")
    axC.set_ylabel("Cumulative total cost (s, log scale)")
    axC.set_title("C. Scaling — PILA pays training once, then ~1 ms/scene\n(dashed flattens; solid keeps climbing — gap widens with stack size)")
    axC.legend(fontsize=7.5, loc="lower right")
    axC.grid(alpha=0.3)

    # Panel D: training-inclusive summary — break-even + full-stack speedup per config
    o2 = timing.sort_values("stack_speedup").reset_index(drop=True)
    yp = np.arange(len(o2))
    bars = axD.barh(yp, o2["stack_speedup"], color=[model_color[m] for m in o2["model"]],
                    edgecolor="k", linewidth=0.4)
    axD.set_yticks(yp)
    axD.set_yticklabels(o2["config"].str.replace("_A", "", regex=False), fontsize=7)
    axD.set_xscale("log")
    axD.set_xlabel("Full-stack speedup INCLUDING training (×, log scale)")
    axD.axvline(1, color="k", lw=1, ls=":")
    axD.set_title(f"D. End-to-end speedup over a full stack (training charged)\nmedian {med_stack:.0f}×; break-even median {med_break:.1f} scenes")
    # annotate break-even scenes on each bar
    for y, (_, r) in zip(yp, o2.iterrows()):
        axD.text(r["stack_speedup"] * 1.05, y, f"BE≈{r['breakeven']:.1f}", va="center", fontsize=6)
    # model legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=model_color[m]) for m in ["Mogi", "Sun69", "Okada"]]
    axD.legend(handles, ["Mogi", "Sun69", "Okada"], loc="lower right", fontsize=8, title="Source model")
    axD.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "PILA vs Classical (SA + Bayesian MCMC) — accuracy parity at ~10⁴× marginal speed, "
        f"~{med_stack:.0f}× end-to-end incl. training\n"
        "18 configs · 3 volcanoes · 3 source models · 2 tracks · CPU timing",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # --- [3/4] Save (png 150 dpi + pdf) ---
    print("[3/4] Saving figure...")
    fig.savefig(OUT_PNG, dpi=150)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    # --- [4/4] Print headline numbers ---
    print("[4/4] Headline numbers:")
    print(f"  marginal speedup (per scene):   median {med_marg:,.0f}×")
    print(f"  training wall-time:             median {med_train:.1f} s/config (all CPU, 150 epochs)")
    print(f"  break-even:                     median {med_break:.1f} scenes")
    print(f"  full-stack speedup incl train:  median {med_stack:.1f}×")
    print(f"Done. Saved {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()
