"""Render the submitted figures directly from the experiment CSV and JSON files."""

import csv
import json
from math import exp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 10.5, "axes.labelsize": 10.5, "axes.titlesize": 11.0,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
COLORS = ("#244f80", "#ac5326", "#42725a", "#774a78", "#635939")
STYLES = ("-", "--", "-.", ":", (0, (5, 2, 1, 2)))
MARKERS = ("o", "s", "^")


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def finish(fig, output, name):
    fig.savefig(output / (name + ".pdf"))
    fig.savefig(output / (name + ".png"), dpi=240)
    plt.close(fig)


def draw_field(data, out):
    field = np.loadtxt(data / "task1_field_n31.csv", delimiter=",")
    radius = (field.shape[0] - 1) // 2
    x = np.arange(field.shape[0]) - radius
    fig, (ax, section) = plt.subplots(1, 2, figsize=(6.18, 2.43),
                                     layout="constrained", width_ratios=[1, 1.16])
    heat = ax.imshow(field, origin="lower", cmap="cividis", vmin=0,
                     extent=(-radius-.5, radius+.5, -radius-.5, radius+.5),
                     interpolation="nearest")
    ax.axhline(0, color="white", alpha=.55, lw=.65, ls="--")
    ax.plot(0, 0, marker="+", color="#fff", markersize=7, markeredgewidth=1.05)
    ax.set(xlabel=r"$x-x_0$ (sites)", ylabel=r"$y-y_0$ (sites)",
           title="(a) Lattice response", xticks=[-16, -8, 0, 8, 16],
           yticks=[-16, -8, 0, 8, 16])
    colorbar = fig.colorbar(heat, ax=ax, fraction=.055, pad=.055, ticks=[0, .1, .2, .3])
    colorbar.set_label(r"$u_{i,j}$")
    section.plot(x, field[radius], "o-", markersize=2.6, color=COLORS[0], lw=1.3)
    section.set(xlabel=r"$x-x_0$ (sites)", ylabel=r"$u_{i,j_0}$",
                title="(b) Central horizontal cut", xlim=(-16, 16), ylim=(0, .405),
                xticks=[-16, -8, 0, 8, 16])
    section.grid(alpha=.19)
    finish(fig, out, "task1_field")


def draw_mass(data, out, summary):
    profiles = read_rows(data / "normalized_profiles.csv")
    fig, (whole, near) = plt.subplots(1, 2, figsize=(6.18, 2.22), layout="constrained")
    for i, case in enumerate(summary["tasks"]["propagation"]):
        m = case["mass"]
        rows = [r for r in profiles if float(r["mass"]) == m]
        x = np.array([int(r["x_offset"]) for r in rows])
        values = np.array([float(r["normalized"]) for r in rows])
        label = fr"$m={m:.1f}$"
        whole.plot(x, values, color=COLORS[i], ls=STYLES[i], lw=1.45, label=label)
        included = (x >= 0) & (x <= 8)
        near.plot(x[included], values[included], color=COLORS[i], ls=STYLES[i], lw=1.2,
                  marker=MARKERS[i], markersize=3.2)
        near.plot(case["interpolated_range"], exp(-1), marker=MARKERS[i], color=COLORS[i],
                  ms=5.5, mfc="white", mew=1.1, zorder=5)
    whole.legend(loc="upper right")
    whole.set(title="(a) Full central cut", xlabel=r"$x-x_0$ (sites)",
              ylabel=r"$u(x,y_0)/u(x_0,y_0)$", xlim=(-32, 32), ylim=(0, 1.04),
              xticks=[-32, -16, 0, 16, 32])
    near.axhline(exp(-1), color=".30", ls=":", lw=.9)
    near.text(7.8, exp(-1)+.02, r"$e^{-1}$", ha="right", va="bottom")
    near.set(title="(b) Threshold crossing", xlabel=r"$r=x-x_0$ (sites)",
             xlim=(0, 8), ylim=(0, 1.04), xticks=range(0, 9, 2))
    for ax in (whole, near):
        ax.grid(alpha=.16)
    finish(fig, out, "mass_response")


def draw_scan(data, out, summary):
    rows = read_rows(data / "omega_scan.csv")
    fig, panels = plt.subplots(2, 2, figsize=(6.18, 4.34), layout="constrained")
    overview = panels[0, 0]
    for i, case in enumerate(summary["tasks"]["omega_scan"]):
        mass = case["mass"]
        matching = [r for r in rows if float(r["mass"]) == mass and r["converged"] == "True"]
        coarse = [r for r in matching if r["stage"] in {"coarse_0.1", "near_2"}]
        overview.semilogy([float(r["omega"]) for r in coarse], [int(r["sweeps"]) for r in coarse],
                          color=COLORS[i], ls=STYLES[i], marker=MARKERS[i], ms=3.3,
                          label=fr"$m={mass:.1f}$", lw=1.1)
        ax = panels.flat[i + 1]
        window = {}
        for r in matching:
            if r["stage"] in {"fine_0.005", "refined_0.001"}:
                window[float(r["omega"])] = int(r["sweeps"])
        ordered = sorted(window)
        ax.plot(ordered, [window[w] for w in ordered], "-", color=COLORS[i], lw=1.05)
        ax.scatter(ordered, [window[w] for w in ordered], color=COLORS[i], marker=MARKERS[i], s=7)
        ax.axvline(case["omega_theory"], color=".32", ls="--", lw=1.05)
        center = case["omega_experimental"]
        ax.scatter(center, case["min_sweeps"], marker="*", s=77,
                   color="black", zorder=6)
        ax.set_title(f"({chr(98+i)}) Local scan, $m={mass:.1f}$")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    overview.legend(loc="upper left")
    overview.set(title="(a) Coarse scan and near 2", xlim=(.98, 2.02))
    for ax in panels.flat:
        ax.set(xlabel=r"$\omega$", ylabel="Full sweeps")
        ax.grid(alpha=.16)
    finish(fig, out, "omega_scan")


def draw_history(data, out):
    rows = read_rows(data / "residual_history.csv")
    fig, (early, extended) = plt.subplots(1, 2, figsize=(6.18, 2.72), layout="constrained")
    series = [
        ("Jacobi", "Jacobi", COLORS[0], "-"),
        ("GS", "Gauss–Seidel", COLORS[1], "--"),
        ("SOR fine", "SOR empirical", COLORS[2], "-."),
        ("SOR theory", "SOR theory", COLORS[3], ":"),
        ("SOR 1.90", r"SOR $\omega=1.90$", COLORS[1], "--"),
        ("SOR 1.99", r"SOR $\omega=1.99$", COLORS[0], "-"),
    ]
    for k, (name, label, color, style) in enumerate(series):
        matched = [r for r in rows if r["label"] == name]
        x = [int(r["sweep"]) for r in matched]
        residual = [float(r["residual"]) for r in matched]
        ax = early if k < 4 else extended
        ax.semilogy(x, residual, ls=style, color=color, lw=1.28, label=label)
    for ax in (early, extended):
        ax.axhline(1e-6, color=".35", ls=(0, (1, 2)), lw=.95)
        ax.set(xlabel="Full sweeps", ylabel=r"$\|r\|_{\infty}$", ylim=(4e-7, 2))
        ax.legend(loc="upper right")
        ax.grid(alpha=.19)
    early.set_title("(a) Method and optimum comparison")
    extended.set_title("(b) Slowdown close to 2")
    finish(fig, out, "convergence")


def draw_boundaries(data, out, summary):
    rows = read_rows(data / "boundary_profiles.csv")
    fig, (ax, offset) = plt.subplots(1, 2, figsize=(6.18, 2.68), layout="constrained")
    for i, n in enumerate((31, 63, 127)):
        match = [r for r in rows if int(r["n"]) == n and int(r["distance"]) <= 32
                 and float(r["normalized"]) > 0]
        ax.semilogy([int(r["distance"]) for r in match], [float(r["normalized"]) for r in match],
                    color=COLORS[i], ls=STYLES[i], marker=MARKERS[i], ms=2.6,
                    markevery=4, label=fr"$N={n}$")
    ax.axvline(16, color=".45", ls=":", lw=.9)
    ax.set(title=r"(a) Domain size, $m=0.2$", xlabel=r"$r$ (sites)", ylabel="Normalized response",
           xlim=(0, 32), ylim=(5e-5, 1.2))
    ax.legend()
    shift = summary["off_center_source"]
    grid = np.loadtxt(data / "offset_source_field.csv", delimiter=",")
    x0, y0 = shift["source"]
    x = np.arange(grid.shape[1]) - x0
    cut = grid[y0] / grid[y0, x0]
    keep = (x >= -5) & (x <= 14)
    offset.plot(x[keep], cut[keep], "o-", lw=1.3, ms=3, color=COLORS[0])
    offset.axvline(-5, color=".45", ls=":", lw=.9)
    offset.set(title=r"(b) Offset source, $m=0.5$", xlabel=r"$x-x_0$ (sites)",
               xlim=(-5.6, 14.5), ylim=(0, 1.04), xticks=[-5, 0, 5, 10])
    offset.text(-4.6, .82, "left boundary", color=".30", ha="left")
    for subplot in (ax, offset):
        subplot.grid(alpha=.2)
    finish(fig, out, "boundaries")


def draw_all(results):
    results = Path(results)
    data, out = results / "data", results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    draw_field(data, out)
    draw_mass(data, out, summary)
    draw_scan(data, out, summary)
    draw_history(data, out)
    draw_boundaries(data, out, summary)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    draw_all(parser.parse_args().output)
