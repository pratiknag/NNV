"""
Section 4: total precipitable water (TPW) data of the MIRS instrument.

Following Section 7 of Jurek and Katzfuss (2022), the 263 x 263 grid over the
continental US (the western 263 columns of Dataset/MIRS.csv) is restricted to
its inner 247 x 247 cells (n = 61,009), and 11 consecutive time points are
used. Two hold-out schemes: "random" holds out 10% of the observations at
each time point at random; "gaps" holds out the observations at time t that
lie in the swath gaps of an unused day (the day whose gaps cover the fraction
of the observations at t closest to --gap-fraction), which mimics realistic
contiguous gaps.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Dataset/MIRS.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("application_mirs"))
    parser.add_argument("--grid-size", type=int, default=263)
    parser.add_argument("--border", type=int, default=8)
    parser.add_argument("--first-day", type=int, default=16)
    parser.add_argument("--n-times", type=int, default=11)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--gap-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2011)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)
    lons = np.sort(raw["lon"].unique())
    lats = np.sort(raw["lat"].unique())
    raw["ix"] = np.searchsorted(lons, raw["lon"])
    raw["iy"] = np.searchsorted(lats, raw["lat"])

    # Inner cells of the western grid_size x grid_size block
    lo, hi = args.border, args.grid_size - args.border
    side = hi - lo
    n = side * side
    days = list(range(args.first_day, args.first_day + args.n_times))
    data = raw[raw["ix"].between(lo, hi - 1) & raw["iy"].between(lo, hi - 1)
               & raw["day"].isin(days)].dropna(subset=["value"])

    # Cell k = row * side + col, with the column (longitude) running fastest,
    # matching advection_diffusion_matrix
    cols = np.arange(side)
    grid = pd.DataFrame({
        "col": np.tile(cols, side),
        "row": np.repeat(cols, side),
    })
    grid["lon"] = lons[lo + grid["col"]]
    grid["lat"] = lats[lo + grid["row"]]
    # Coordinates on the unit square: the grid is regular in the Mercator
    # projection, so grid-index distances are proportional to map distances
    grid["x1"] = grid["col"] / side
    grid["x2"] = grid["row"] / side
    grid.to_csv(args.output_dir / "grid.csv", index=False)
    grid[["x1", "x2"]].to_csv(args.output_dir / "locs.csv", index=False)

    rng = np.random.default_rng(args.seed)
    Y = np.full((args.n_times, n), np.nan)
    split = np.zeros((args.n_times, n), dtype=np.int8)  # 0 none, 1 train, 2 test
    for t, day in enumerate(days):
        day_data = data[data["day"] == day]
        cells = ((day_data["iy"] - lo) * side + (day_data["ix"] - lo)).to_numpy()
        Y[t, cells] = day_data["value"].to_numpy()
        test = rng.random(cells.size) < args.test_fraction
        split[t, cells] = np.where(test, 2, 1)
    # Swath masks of the unused days
    def cells_of(frame):
        return ((frame["iy"] - lo) * side + (frame["ix"] - lo)).to_numpy()
    other = raw[raw["ix"].between(lo, hi - 1) & raw["iy"].between(lo, hi - 1)
                & ~raw["day"].isin(days)].dropna(subset=["value"])
    masks = {}
    for day, frame in other.groupby("day"):
        masks[day] = np.zeros(n, bool)
        masks[day][cells_of(frame)] = True
    split_gaps = np.zeros_like(split)
    gap_days = []
    for t in range(args.n_times):
        obs = split[t] > 0
        frac = {day: (obs & ~m).sum() / obs.sum() for day, m in masks.items()}
        day = min(frac, key=lambda k: abs(frac[k] - args.gap_fraction))
        gap_days.append(day)
        split_gaps[t, obs] = np.where(masks[day][obs], 1, 2)
    np.savez_compressed(args.output_dir / "data.npz", Y=Y, split=split,
                        split_gaps=split_gaps, gap_days=np.array(gap_days),
                        days=np.array(days), side=side)
    summary = pd.DataFrame({
        "t": np.arange(1, args.n_times + 1),
        "day": days,
        "n_obs": (split > 0).sum(1),
        "n_train": (split == 1).sum(1),
        "n_test": (split == 2).sum(1),
        "gap_day": gap_days,
        "n_test_gaps": (split_gaps == 2).sum(1),
    })
    summary.to_csv(args.output_dir / "data_summary.csv", index=False)
    print(f"n = {n} ({side} x {side}), lon {grid['lon'].min():.2f} to "
          f"{grid['lon'].max():.2f}, lat {grid['lat'].min():.2f} to "
          f"{grid['lat'].max():.2f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
