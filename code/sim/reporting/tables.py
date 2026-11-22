"""Paper tables (Table 1 and Table 2) as CSV and booktabs LaTeX, and `build_all`, which turns the
result folders of the experiments into every table and figure of the paper.

Methodology map (03_methodology.md): section 7 (outputs).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from .. import config as config_mod
from .. import methods as methods_mod

log = logging.getLogger(__name__)

_ESCAPES = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}


def tex_escape(text) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in str(text))


def format_cell(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return tex_escape(value)


def to_latex(df: pd.DataFrame, caption: str = "", label: str = "", digits: int = 3,
             bold_max: list = None, bold_min: list = None) -> str:
    """A booktabs table. Columns listed in `bold_max` / `bold_min` get their best entry in bold."""
    bold = {}
    for cols, pick in ((bold_max or [], "idxmax"), (bold_min or [], "idxmin")):
        for col in cols:
            if col in df.columns and df[col].notna().any():
                bold[col] = getattr(df[col].astype(float), pick)()
    align = "".join("l" if df[c].dtype == object else "r" for c in df.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        cells = []
        for col in df.columns:
            text = format_cell(row[col], digits)
            cells.append(rf"\textbf{{{text}}}" if bold.get(col) == idx else text)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def table1(tab1: pd.DataFrame, objectives: list) -> pd.DataFrame:
    """Table 1: one row per simulator."""
    cols = ["simulator", "accuracy", "macro_f1", "brier", "ece", "sigma_bias"]
    cols += [f"regret_{o}" for o in objectives if f"regret_{o}" in tab1.columns]
    out = tab1[[c for c in cols if c in tab1.columns]].copy()
    return out.rename(columns={"macro_f1": "macro-F1", "sigma_bias": "bias"})


def table2(tab2: pd.DataFrame, simulator: str, rho: float, objective: str, certificate: str) -> pd.DataFrame:
    """Table 2: one row per decision rule for a given simulator, pilot fraction and objective."""
    d = tab2[(tab2["simulator"] == simulator) & np.isclose(tab2["rho"], rho) & (tab2["objective"] == objective)]
    cols = {"method": "method", "true_welfare": "true welfare", "regret": "regret",
            "true_min_group": "worst-group welfare", f"coverage_{certificate}": "coverage",
            f"cert_{certificate}": "certificate", f"slack_{certificate}": "slack"}
    out = d[[c for c in cols if c in d.columns]].rename(columns=cols).copy()
    out["method"] = out["method"].map(lambda m: methods_mod.METHOD_LABELS.get(m, m))
    return out.reset_index(drop=True)


def _read(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path, encoding="utf-8") if os.path.exists(path) else None


def build_all(cfg, out: str = None) -> list:
    """Create every table and figure for which results exist. Returns the files written."""
    from . import figures

    results = os.path.join(config_mod.resolve_path(cfg, cfg["paths"]["results_dir"]), cfg["testbed"]["name"])
    out_dir = out or os.path.join(results, "paper")
    os.makedirs(out_dir, exist_ok=True)
    save_kw = {"formats": tuple(cfg["figures"]["formats"]), "dpi": int(cfg["figures"]["dpi"])}
    objectives = list(cfg["welfare"]["objectives"])
    main_cert = cfg["guard"]["ucb"]
    written = []

    def write_table(df: pd.DataFrame, name: str, **latex_kw) -> None:
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, encoding="utf-8")
        with open(os.path.join(out_dir, f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(to_latex(df, **latex_kw))
        written.extend([os.path.join(out_dir, f"{name}.csv"), os.path.join(out_dir, f"{name}.tex")])

    tab1 = _read(os.path.join(results, "rq1", "tab1_simulators.csv"))
    if tab1 is not None:
        write_table(table1(tab1, objectives), "tab1_simulators", caption="Simulator validity.",
                    label="tab:simulators", bold_max=["accuracy", "macro-F1"], bold_min=["brier", "ece"])
        bias = _read(os.path.join(results, "rq1", "bias_matrix.csv"))
        if bias is not None:
            references = set(tab1.loc[tab1["is_reference"].astype(bool), "simulator"]) | {"uniform"}
            for sim in bias["simulator"].unique():
                if sim not in references:  # reference rows have no structure worth a heat map
                    written += figures.bias_heatmap(bias, out_dir, sim, **save_kw)
    curves = _read(os.path.join(results, "rq2", "pressure_curve.csv"))
    if curves is not None:
        written += figures.fig2_goodhart_curves(curves, out_dir, **save_kw)
    plugin = _read(os.path.join(results, "rq2", "plugin_summary.csv"))
    if plugin is not None:
        written += figures.fig3_group_decomposition(plugin, out_dir, **save_kw)
    tab2 = _read(os.path.join(results, "rq3", "tab2_methods.csv"))
    if tab2 is not None:
        for sim in tab2["simulator"].unique():
            written += figures.fig4_pilot_size(tab2, out_dir, sim, certificate=main_cert,
                                               name=f"fig4_pilot_size_{sim}", **save_kw)
            for rho in sorted(tab2["rho"].unique()):
                for obj in objectives:
                    t = table2(tab2, sim, float(rho), obj, main_cert)
                    if not t.empty:
                        write_table(t, f"tab2_{sim}_rho{rho:g}_{obj}", label=f"tab:methods-{obj}",
                                    caption=f"Decision rules ({tex_escape(sim)}, pilot fraction {rho:g}, {obj}).",
                                    bold_max=["true welfare", "worst-group welfare"], bold_min=["regret"])
    summary = _read(os.path.join(results, "synthetic", "summary.csv"))
    if summary is not None and (summary["scenario"] == "structured").any():
        for obj in objectives:
            written += figures.fig5_structured_misreading(summary, out_dir, obj,
                                                          name=f"fig5a_structured_misreading_{obj}", **save_kw)
        coverage = _read(os.path.join(results, "synthetic", "coverage.csv"))
        if coverage is not None:
            for obj in objectives:
                try:
                    written += figures.fig5_certificate_coverage(coverage, out_dir, obj,
                                                                 name=f"fig5b_certificate_coverage_{obj}", **save_kw)
                except ValueError as err:
                    log.warning("fig5b (%s) skipped: %s", obj, err)
    if not written:
        log.warning("no results found under %s: run the experiments first", results)
    return written
