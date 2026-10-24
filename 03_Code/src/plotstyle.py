"""Shared figure style: dark text, recessive grid, validated categorical palette, 300 dpi."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#3a3936", "#d9d8d4"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS = "#d95926", "#f2f1ee", "#1c5cab"


def apply():
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
        "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
        "axes.facecolor": "white", "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    })


def save(fig, path):
    fig.savefig(path, facecolor="white")
    # Vector copy for submission (Elsevier asks for EPS/PDF or >= 1000 dpi for line art).
    fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor="white")
    plt.close(fig)
