"""Shared figure style: one palette, one set of mark specifications.

Colour is assigned by the job it does:
  categorical  series identity. Eight hues in a fixed order; the first two/three are also safe when
               any two marks may sit side by side. The palette passes the lightness, chroma,
               colour-vision-deficiency and normal-vision separation checks on a white surface;
               aqua, yellow and magenta are below 3:1 contrast, so every chart that uses them also
               carries marker shapes, a legend and a CSV of the plotted values.
  diverging    signed quantities (bias): blue <-> red around a neutral grey midpoint.
  emphasis     one accent series against grey context.
A method keeps its colour and marker in every figure ("colour follows the entity").
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#b5b3ab"       # de-emphasised marks (context series, reference lines)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
NEUTRAL_MID = "#f0efec"

DIVERGING = LinearSegmentedColormap.from_list("sim_diverging", [BLUE, NEUTRAL_MID, RED])

LINE_WIDTH = 1.7
MARKER_SIZE = 5.5
MARKER_RING = 1.1

# (colour, marker, label): fixed per decision rule across all figures
METHOD_STYLE = {
    "guard": (BLUE, "o", "GUARD (rectify)"),
    "sim_plugin": (ORANGE, "s", "Simulator plug-in"),
    "pilot_sim_plugin": (AQUA, "^", "Pilot + simulator plug-in"),
    "guard_pess": (YELLOW, "D", "GUARD-pess"),
    "guard_est": (MAGENTA, "v", "GUARD-est"),
    "pilot_only": (GREEN, "P", "Pilot only"),
    "ens_lcb": (VIOLET, "X", "Ensemble-LCB"),
    "hedgetune": (RED, "*", "HedgeTune"),
}
# reference levels are context, not series: two greys told apart by lightness and by their labels
REFERENCE_STYLE = {"oracle": (INK_SECONDARY, "Oracle"), "ed": (CONTEXT, "ED (retain all)")}

QUANTITY_STYLE = {
    "sim": (ORANGE, "s", "Simulated welfare"),
    "true": (BLUE, "o", "True welfare"),
}
CERTIFICATE_STYLE = {
    "betting": (BLUE, "o", "Group-conditional"),
    "pooled": (ORANGE, "s", "Pooled (no groups)"),
    "count": (AQUA, "^", "Count-based"),
    "hoeffding_serfling": (YELLOW, "D", "Group-conditional (H-S)"),
}
OBJECTIVE_LABELS = {"utilitarian": "Utilitarian (p = 1)", "nash": "Nash (p = 0)", "maxmin": "Max-min (p = -inf)"}


def apply_style() -> None:
    """Quiet chrome, thin marks; the data is the only loud thing."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.major.size": 2.5, "ytick.major.size": 0, "xtick.major.width": 0.7,
        "lines.linewidth": LINE_WIDTH, "lines.markersize": MARKER_SIZE, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


def line_kwargs(colour: str, marker: str) -> dict:
    """A 2px line with ring-separated markers (the ring keeps markers legible where lines cross)."""
    return {"color": colour, "marker": marker, "linewidth": LINE_WIDTH, "markersize": MARKER_SIZE,
            "markeredgecolor": SURFACE, "markeredgewidth": MARKER_RING}


def text_on(fill_rgb) -> str:
    """White or near-black label colour for text placed inside a coloured fill."""
    r, g, b = fill_rgb[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if luminance > 0.55 else "#ffffff"
