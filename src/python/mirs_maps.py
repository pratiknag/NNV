"""
Section 4 maps at t = 1, 6 and 11 (random hold-outs), from the filtering
fields in application_mirs/results/ (mirs_validate7.py --save-fields):
  figures/application-maps.pdf     Figure 8: data (including the held-out
                                   values), NNV and HV filtering means, and the
                                   widths of the pixelwise 95% prediction
                                   intervals, with zoom insets where the HV
                                   widths jump at its partition lines
  figures/application-lowrank.pdf  Figure S7: the same for low-rank
"""
import json
import numpy as np
import pandas as pd
import mirs_app_plot as m
from plot_results import FIGURE_DIR
from plot_sparsity import STYLE

D = m.DATA / "results"
# (r0, r1, c0, c1, inset corner): regions with HV-only width jumps at partition lines
ZOOMS = {1: (173, 197, 60, 84, (0.42, 0.03)),
         6: (172, 198, 110, 136, (0.03, 0.03)),
         11: (172, 198, 171, 197, (0.03, 0.03))}


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(m.DATA / "data.npz")
    side = int(data["side"])
    grid = pd.read_csv(m.DATA / "grid.csv")
    noise = json.loads((D / "params_used.json").read_text())["noise"]
    f = {k: np.load(D / f"fields_random_{k}.npz") for k in ["NNV", "HV", "LR"]}
    widths = {k: 2 * 1.96 * np.sqrt(v["sd"].astype(float) ** 2 + noise)
              for k, v in f.items()}
    labels = [STYLE[k][0] for k in ["NNV", "HV"]]
    m.means_and_widths(data["Y"], [f[k]["mean"] for k in ["NNV", "HV"]],
                       [widths[k] for k in ["NNV", "HV"]], grid, side, labels,
                       FIGURE_DIR / "application-maps.pdf", hv_lines=(labels[1],),
                       width_step=1.0, zooms=ZOOMS, zoom_methods=tuple(labels[::-1]))
    m.means_and_widths(data["Y"], [f["LR"]["mean"]], [widths["LR"]], grid, side,
                       [STYLE["LR"][0]], FIGURE_DIR / "application-lowrank.pdf",
                       panel=(3.4, 3.2), width_step=1.0)
    print("wrote", FIGURE_DIR / "application-maps.pdf", "and",
          FIGURE_DIR / "application-lowrank.pdf")


if __name__ == "__main__":
    main()
