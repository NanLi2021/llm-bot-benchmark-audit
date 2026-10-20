"""Shared publication styling for the paper's figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SOFT, GRID = "#222222", "#444444", "#D9D9D9"


def publication_style():
    """Times-compatible serif text without requiring an external TeX renderer."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "text.color": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig, name, out_dir):
    """Keep the physical canvas size so fonts retain their size in LaTeX."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{extension}", dpi=300)
        print("wrote", out_dir / f"{name}.{extension}")
    plt.close(fig)
