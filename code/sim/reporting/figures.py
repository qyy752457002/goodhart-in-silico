"""Figures of the paper, drawn from the experiments' result tables.

  fig2_goodhart_curves        Fig. 2  simulated vs true welfare against optimisation pressure
  fig3_group_decomposition    Fig. 3  per-group overestimate at the plug-in design
  fig4_pilot_size             Fig. 4  certificate and true welfare against the pilot fraction
  fig5_structured_misreading  Fig. 5a decision rules under a structured misreading
  fig5_certificate_coverage   Fig. 5b coverage of pooled vs group-conditional certificates
  bias_heatmap                Table 1 companion: group x proposal signed support error

Every figure is written as PDF and PNG together with a CSV of exactly the values it shows.
Methodology map (03_methodology.md): section 7 (outputs).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import style
from .style import plt


def _save(fig, data: pd.DataFrame, out_dir: str, name: str, formats=("pdf", "png"), dpi: int = 200) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in formats:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    data.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, encoding="utf-8")
    plt.close(fig)
    return paths


def _label_line_ends(ax, items: list, min_gap_px: float = 9.0) -> None:
    """Direct labels at the right end of lines. `items` = [(x, y, text)] in priority order; a label
    that would collide with one already placed is dropped (the legend still identifies it)."""
    placed = []
    for x, y, text in items:
        if not np.isfinite(y):
            continue
        y_px = ax.transData.transform((x, y))[1]
        if any(abs(y_px - other) < min_gap_px for other in placed):
            continue
        placed.append(y_px)
        ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=7, color=style.INK_SECONDARY, annotation_clip=False)


def _figure_legend(fig, handles: list, labels: list, ncol: int = None) -> None:
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol or len(labels),
               handlelength=2.2, columnspacing=1.6, borderaxespad=0.2)


def _pick(values, requested, what: str) -> list:
    """`requested` filtered to what exists (keeping its order); None means everything available."""
    available = list(dict.fromkeys(values))
    if requested is None:
        return available
    chosen = [v for v in requested if v in available]
    if not chosen:
        raise ValueError(f"none of the requested {what} {list(requested)} is in the results (have {available})")
    return chosen


# ----------------------------------------------------------------------------------------------
# Fig. 2
# ----------------------------------------------------------------------------------------------

def fig2_goodhart_curves(curves: pd.DataFrame, out_dir: str, simulators: list = None, objectives: list = None,
                         name: str = "fig2_goodhart_curves", **save_kw) -> list:
    """Small multiples (simulators x objectives). Each panel: expected simulated welfare of the
    best-of-N package and its true welfare; the widening distance is the Goodhart gap."""
    style.apply_style()
    sims = _pick(curves["simulator"], simulators, "simulators")
    objs = _pick(curves["objective"], objectives, "objectives")
    fig, axes = plt.subplots(len(sims), len(objs), figsize=(2.35 * len(objs) + 0.6, 1.75 * len(sims) + 0.35),
                             sharex=True, squeeze=False, layout="constrained")
    shown, handles = [], {}
    for r, sim in enumerate(sims):
        for c, obj in enumerate(objs):
            ax = axes[r, c]
            d = curves[(curves["simulator"] == sim) & (curves["objective"] == obj)].sort_values("log2_n")
            if d.empty:
                ax.set_axis_off()
                continue
            x = d["log2_n"].to_numpy()
            ends = []
            for key in ("sim", "true"):
                colour, marker, label = style.QUANTITY_STYLE[key]
                y = d[f"{key}_exact"].to_numpy()
                if f"{key}_lo" in d.columns:
                    ax.fill_between(x, d[f"{key}_lo"], d[f"{key}_hi"], color=colour, alpha=0.12, linewidth=0)
                (handles[label],) = ax.plot(x, y, label=label, **style.line_kwargs(colour, marker))
                ends.append((x[-1], y[-1], label.split()[0]))
            gap = float(d["overestimate_exact"].iloc[-1])
            ax.text(0.97, 0.05, f"gap at full pressure {gap:+.3f}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=7, color=style.INK_SECONDARY)
            if c == len(objs) - 1:
                fig.canvas.draw()
                _label_line_ends(ax, ends)
            if r == 0:
                ax.set_title(style.OBJECTIVE_LABELS.get(obj, obj))
            if c == 0:
                ax.set_ylabel(f"{sim}\nwelfare")
            ax.set_xticks([t for t in x if t % 4 == 0 or t == x[-1]])
            shown.append(d.assign(panel_row=sim, panel_col=obj))
    fig.supxlabel("Optimisation pressure: log2 N (best of N random packages; right end = exact optimum)",
                  fontsize=8, color=style.INK_SECONDARY)
    _figure_legend(fig, list(handles.values()), list(handles))
    cols = ["simulator", "objective", "log2_n", "n", "sim_exact", "true_exact", "overestimate_exact"]
    cols += [c for c in ("sim_lo", "sim_hi", "true_lo", "true_hi") if c in curves.columns]
    return _save(fig, pd.concat(shown, ignore_index=True)[cols], out_dir, name, **save_kw)


# ----------------------------------------------------------------------------------------------
# Fig. 3
# ----------------------------------------------------------------------------------------------

def fig3_group_decomposition(plugin: pd.DataFrame, out_dir: str, simulators: list = None,
                             objectives: list = None, name: str = "fig3_group_decomposition", **save_kw) -> list:
    """Per-group overestimate delta_g = simulated - true group welfare at the plug-in design.
    The truly worst-off group is drawn in the accent colour, the others in grey: illusory fairness
    is that group standing far to the right."""
    style.apply_style()
    sims = _pick(plugin["simulator"], simulators, "simulators")
    objs = _pick(plugin["objective"], objectives, "objectives")
    groups = [c[len("delta["):-1] for c in plugin.columns if c.startswith("delta[")]
    fig, axes = plt.subplots(len(sims), len(objs), figsize=(2.5 * len(objs) + 1.2, 0.34 * len(groups) * len(sims)
                                                             + 0.75 * len(sims) + 0.5),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    y = np.arange(len(groups))[::-1]
    rows = []
    for r, sim in enumerate(sims):
        for c, obj in enumerate(objs):
            ax = axes[r, c]
            d = plugin[(plugin["simulator"] == sim) & (plugin["objective"] == obj)]
            if d.empty:
                ax.set_axis_off()
                continue
            rec = d.iloc[0]
            values = np.array([rec[f"delta[{g}]"] for g in groups], dtype=float)
            worst = rec["true_worst_group"]
            colours = [style.BLUE if g == worst else style.CONTEXT for g in groups]
            ax.axvline(0, color=style.AXIS, linewidth=0.8, zorder=1)
            ax.hlines(y, 0, values, colors=colours, linewidth=2.2, zorder=2)
            ax.scatter(values, y, s=42, c=colours, edgecolors=style.SURFACE, linewidths=style.MARKER_RING, zorder=3)
            for yi, v in zip(y, values):
                ax.annotate(f"{v:+.3f}", (v, yi), xytext=(6 if v >= 0 else -6, 0), textcoords="offset points",
                            ha="left" if v >= 0 else "right", va="center", fontsize=7, color=style.INK_SECONDARY)
            ax.set_yticks(y, groups)
            ax.set_ylim(-0.6, len(groups) - 0.4)
            ax.grid(False)
            ax.margins(x=0.28)
            title = style.OBJECTIVE_LABELS.get(obj, obj) if r == 0 else ""
            ax.set_title(f"{title}\nIF = {rec['illusory_fairness']:+.3f}, regret = {rec['regret']:.3f}".strip(),
                         fontsize=7.5)
            if c == 0 and len(sims) > 1:
                ax.set_ylabel(sim)
            rows += [{"simulator": sim, "objective": obj, "group": g, "delta": v, "is_true_worst": g == worst,
                      "illusory_fairness": rec["illusory_fairness"], "regret": rec["regret"]}
                     for g, v in zip(groups, values)]
    fig.supxlabel("Overestimate of group welfare at the selected package (simulated - true)", fontsize=8,
                  color=style.INK_SECONDARY)
    handles = [plt.Line2D([], [], color=col, marker="o", linewidth=2.2, markersize=6, markeredgecolor=style.SURFACE)
               for col in (style.BLUE, style.CONTEXT)]
    _figure_legend(fig, handles, ["Truly worst-off group", "Other groups"])
    return _save(fig, pd.DataFrame(rows), out_dir, name, **save_kw)


# ----------------------------------------------------------------------------------------------
# Fig. 4
# ----------------------------------------------------------------------------------------------

def fig4_pilot_size(tab2: pd.DataFrame, out_dir: str, simulator: str = None, objectives: list = None,
                    methods: tuple = ("guard", "pilot_only", "sim_plugin", "pilot_sim_plugin"),
                    certificate: str = "betting", name: str = "fig4_pilot_size", **save_kw) -> list:
    """True welfare of the selected package and GUARD's certificate as the pilot grows. The shaded
    band between GUARD's true welfare and its certificate is the certificate's slack. The pilot-only
    rule is drawn too: the distance from it to GUARD is what the simulator adds once it is rectified."""
    style.apply_style()
    simulator = simulator or tab2["simulator"].iloc[0]
    data = tab2[tab2["simulator"] == simulator]
    objs = _pick(data["objective"], objectives, "objectives")
    methods = [m for m in methods if m in set(data["method"])]
    if not methods:
        raise ValueError("none of the requested methods is in the results")
    fig, axes = plt.subplots(1, len(objs), figsize=(2.55 * len(objs) + 0.5, 2.45), squeeze=False,
                             layout="constrained")
    handles, rows = {}, []
    for c, obj in enumerate(objs):
        ax = axes[0, c]
        d = data[data["objective"] == obj]
        rhos = np.sort(d["rho"].unique())
        ends, reference_ends = [], []
        for ref in ("oracle", "ed"):
            ref_rows = d[d["method"] == ref]
            if not ref_rows.empty:
                colour, label = style.REFERENCE_STYLE[ref]
                level = float(ref_rows["true_welfare"].mean())
                handles[label] = ax.axhline(level, color=colour, linewidth=1.0, zorder=1, label=label)
                reference_ends.append((rhos[-1], level, label.split(" (")[0]))
        for method in methods:
            colour, marker, label = style.METHOD_STYLE[method]
            m = d[d["method"] == method].sort_values("rho")
            kw = {"zorder": 3, **style.line_kwargs(colour, marker)}
            if method == "pilot_only" and "guard" in methods:
                # it often coincides with GUARD: a wide band underneath keeps both readable
                kw.update(zorder=2, linewidth=style.LINE_WIDTH * 3.2, alpha=0.4, marker=None,
                          solid_capstyle="butt")
            (handles[label],) = ax.plot(m["rho"], m["true_welfare"], label=label, **kw)
            ends.append((m["rho"].iloc[-1], m["true_welfare"].iloc[-1], label.split(" (")[0]))
            rows.append(m[["simulator", "objective", "rho", "method", "true_welfare"]].assign(series="true welfare"))
        cert_col = f"cert_{certificate}"
        g = d[d["method"] == "guard"].sort_values("rho")
        if not g.empty and cert_col in g.columns:
            label = "GUARD certificate (lower bound)"
            (handles[label],) = ax.plot(g["rho"], g[cert_col], color=style.BLUE, linestyle=(0, (3, 2)),
                                        linewidth=style.LINE_WIDTH, marker="o", markersize=style.MARKER_SIZE,
                                        markerfacecolor=style.SURFACE, markeredgecolor=style.BLUE,
                                        markeredgewidth=1.2, label=label, zorder=3)
            ax.fill_between(g["rho"], g[cert_col], g["true_welfare"], color=style.BLUE, alpha=0.10, linewidth=0)
            ends.append((g["rho"].iloc[-1], g[cert_col].iloc[-1], "certificate"))
            rows.append(g[["simulator", "objective", "rho", "method"]].assign(true_welfare=g[cert_col],
                                                                             series="certificate"))
        ax.set_title(style.OBJECTIVE_LABELS.get(obj, obj))
        ax.set_xticks(rhos)
        ax.margins(x=0.08)
        if c == 0:
            ax.set_ylabel("Welfare of the selected package")
        if len(ends) <= 4:
            fig.canvas.draw()
            _label_line_ends(ax, ends + reference_ends)   # method lines first: they win collisions
    fig.supxlabel("Pilot fraction per group", fontsize=8, color=style.INK_SECONDARY)
    _figure_legend(fig, list(handles.values()), list(handles), ncol=min(3, len(handles)))
    table = pd.concat(rows, ignore_index=True).rename(columns={"true_welfare": "value"})
    return _save(fig, table, out_dir, name, **save_kw)


# ----------------------------------------------------------------------------------------------
# Fig. 5
# ----------------------------------------------------------------------------------------------

def fig5_structured_misreading(summary: pd.DataFrame, out_dir: str, objective: str = "maxmin", rho: float = None,
                               methods: tuple = ("guard", "sim_plugin", "pilot_sim_plugin", "guard_pess",
                                                 "guard_est"),
                               scenario: str = "structured", name: str = "fig5a_structured_misreading",
                               **save_kw) -> list:
    """True welfare of each decision rule as the misreading of one group gets stronger; one panel
    per misread group (Proposition C: pessimism alone does not repair a structured misreading)."""
    style.apply_style()
    d = summary[(summary["scenario"] == scenario) & (summary["objective"] == objective)]
    if d.empty:
        raise ValueError(f"no rows for scenario {scenario!r} and objective {objective!r}")
    rho = float(rho if rho is not None else np.sort(d["rho"].unique())[0])
    d = d[np.isclose(d["rho"], rho)]
    groups = list(dict.fromkeys(d["group_name"]))
    methods = [m for m in methods if m in set(d["method"])]
    fig, axes = plt.subplots(1, len(groups), figsize=(2.3 * len(groups) + 0.6, 2.5), sharey=True, squeeze=False,
                             layout="constrained")
    handles, rows = {}, []
    for c, group in enumerate(groups):
        ax = axes[0, c]
        dg = d[d["group_name"] == group]
        xs = np.sort(dg["e_s"].unique())
        oracle = dg[dg["method"] == "oracle"]
        if not oracle.empty:
            level = float(oracle["true_welfare"].mean())
            ax.axhline(level, color=style.CONTEXT, linewidth=1.0, zorder=1)
            ax.annotate("Oracle", (xs[0], level), xytext=(2, 3), textcoords="offset points", fontsize=7,
                        color=style.INK_MUTED, va="bottom", ha="left")
        for method in methods:
            colour, marker, label = style.METHOD_STYLE[method]
            m = dg[dg["method"] == method].sort_values("e_s")
            (handles[label],) = ax.plot(m["e_s"], m["true_welfare"], label=label, zorder=3,
                                        **style.line_kwargs(colour, marker))
            rows.append(m[["scenario", "group_name", "e_s", "rho", "objective", "method", "true_welfare"]])
        ax.set_title(f"Misread group: {group}", fontsize=7.5)
        ax.set_xticks(xs)
        ax.margins(x=0.08)
        if c == 0:
            ax.set_ylabel(f"True welfare ({style.OBJECTIVE_LABELS.get(objective, objective)})")
    fig.supxlabel(f"Strength of the structured misreading e_s (pilot fraction {rho:g})", fontsize=8,
                  color=style.INK_SECONDARY)
    _figure_legend(fig, list(handles.values()), list(handles), ncol=min(5, len(handles)))
    return _save(fig, pd.concat(rows, ignore_index=True), out_dir, name, **save_kw)


def fig5_certificate_coverage(coverage: pd.DataFrame, out_dir: str, objective: str = "maxmin", rho: float = None,
                              method: str = "guard", certificates: tuple = ("betting", "pooled"),
                              scenario: str = "structured", name: str = "fig5b_certificate_coverage",
                              **save_kw) -> list:
    """Coverage of the certificate at GUARD's selection, averaged over misread groups: calibrating
    without groups loses coverage as the misreading grows (Proposition B)."""
    style.apply_style()
    d = coverage[(coverage["scenario"] == scenario) & (coverage["objective"] == objective)
                 & (coverage["method"] == method) & coverage["certificate"].isin(certificates)]
    if d.empty:
        raise ValueError("no coverage rows match the request")
    rho = float(rho if rho is not None else np.sort(d["rho"].unique())[0])
    d = d[np.isclose(d["rho"], rho)]
    agg = d.groupby(["certificate", "e_s"], as_index=False).agg(coverage=("coverage", "mean"),
                                                                 nominal=("nominal", "first"))
    fig, ax = plt.subplots(figsize=(4.0, 2.5), layout="constrained")
    nominal = float(agg["nominal"].iloc[0])
    ax.axhline(nominal, color=style.CONTEXT, linewidth=1.0, zorder=1)
    xs = np.sort(agg["e_s"].unique())
    ax.annotate(f"Nominal {nominal:.0%}", (xs[0], nominal), xytext=(2, -3), textcoords="offset points",
                fontsize=7, color=style.INK_MUTED, va="top", ha="left")
    handles, ends = {}, []
    present = [k for k in certificates if k in set(agg["certificate"])]
    for j, kind in enumerate(present):
        colour, marker, label = style.CERTIFICATE_STYLE[kind]
        m = agg[agg["certificate"] == kind].sort_values("e_s")
        kw = style.line_kwargs(colour, marker)
        # earlier series are drawn wider and with larger markers: where two series coincide the
        # one underneath still shows as an outline
        kw["linewidth"] += 1.3 * (len(present) - 1 - j)
        kw["markersize"] += 2.2 * (len(present) - 1 - j)
        (handles[label],) = ax.plot(m["e_s"], m["coverage"], label=label, zorder=3 + j, **kw)
        ends.append((m["e_s"].iloc[-1], m["coverage"].iloc[-1], label.split(" (")[0]))
    ax.set_xticks(xs)
    ax.set_ylim(min(0.0, float(agg["coverage"].min()) - 0.05), 1.06)
    ax.margins(x=0.08)
    ax.set_ylabel("Coverage of the certificate")
    ax.set_xlabel(f"Misreading strength e_s (pilot fraction {rho:g})")
    fig.canvas.draw()
    _label_line_ends(ax, ends)
    _figure_legend(fig, list(handles.values()), list(handles))
    return _save(fig, agg, out_dir, name, **save_kw)


# ----------------------------------------------------------------------------------------------
# Bias heat map
# ----------------------------------------------------------------------------------------------

def bias_heatmap(bias: pd.DataFrame, out_dir: str, simulator: str = None, annotate_above: float = 0.2,
                 name: str = None, **save_kw) -> list:
    """Signed support error (simulated - true) per group and proposal. Red: the simulator
    overstates support; blue: it understates it. Only large cells are annotated."""
    style.apply_style()
    simulator = simulator or bias["simulator"].iloc[0]
    d = bias[bias["simulator"] == simulator]
    grid = d.pivot(index="group", columns="decision", values="bias")
    counts = d.pivot(index="group", columns="decision", values="n_cells")
    limit = max(0.05, float(np.nanmax(np.abs(grid.to_numpy()))))
    fig, ax = plt.subplots(figsize=(0.36 * grid.shape[1] + 2.4, 0.36 * grid.shape[0] + 1.0), layout="constrained")
    mesh = ax.pcolormesh(grid.to_numpy(), cmap=style.DIVERGING, vmin=-limit, vmax=limit, edgecolors=style.SURFACE,
                         linewidth=1.5)
    ax.set_xticks(np.arange(grid.shape[1]) + 0.5, [f"Q{q}" for q in grid.columns])
    ax.set_yticks(np.arange(grid.shape[0]) + 0.5, grid.index)
    ax.invert_yaxis()
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            if np.isfinite(v) and abs(v) >= annotate_above:
                ax.text(j + 0.5, i + 0.5, f"{v:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=6.5,
                        color=style.text_on(mesh.cmap(mesh.norm(v))))
    bar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Simulated - true support", fontsize=7.5)
    bar.outline.set_visible(False)
    ax.set_title(f"{simulator}: signed support error by group and proposal", fontsize=8.5, loc="left")
    table = grid.reset_index().melt(id_vars="group", var_name="decision", value_name="bias")
    table = table.merge(counts.reset_index().melt(id_vars="group", var_name="decision", value_name="n_cells"),
                        on=["group", "decision"])
    return _save(fig, table.assign(simulator=simulator), out_dir, name or f"bias_heatmap_{simulator}", **save_kw)
