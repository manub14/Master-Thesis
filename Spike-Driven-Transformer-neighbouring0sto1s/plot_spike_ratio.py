#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_json_block(text: str, label: str) -> dict:
    """
    Find JSON object that appears right after '<label>:' using brace counting.
    Returns parsed dict.
    """
    # Find the LAST occurrence to catch the final summary if it appears multiple times
    matches = list(re.finditer(rf"{re.escape(label)}\s*:\s*\{{", text))
    if not matches:
        raise ValueError(f"Label '{label}' not found in log.")
    m = matches[-1]
    start = m.end() - 1  # position at '{'
    brace = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError(f"JSON block after '{label}' seems incomplete.")
    block = text[start:end]
    return json.loads(block)

def load_spike_table(log_path: Path) -> pd.DataFrame:
    raw = log_path.read_text(encoding="utf-8", errors="ignore")
    # Try 'firing_rate' first; fall back to 'non_zero'
    try:
        fr = extract_json_block(raw, "firing_rate")
    except Exception:
        fr = extract_json_block(raw, "non_zero")

    rows = []
    for t_key, layers in fr.items():
        for layer, val in layers.items():
            rows.append({"time_step": t_key, "layer": layer, "spike_ratio": float(val)})
    df = pd.DataFrame(rows)
    # Make time steps sortable: t0, t1, ...
    df["t_index"] = df["time_step"].str.extract(r"t(\d+)").astype(int)
    df = df.sort_values(["t_index", "layer"]).reset_index(drop=True)
    return df

def plot_over_time(by_t: pd.DataFrame, outdir: Path):
    x = by_t["t_index"].to_numpy()
    y = by_t["mean"].to_numpy()
    y_std = by_t["std"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)
    plt.xticks(x, [f"t{i}" for i in x])
    plt.xlabel("Time step")
    plt.ylabel("Spike ratio (mean across layers)")
    plt.title("Network Spike Ratio over Time (mean +/- 1 SD)")
    plt.tight_layout()
    out = outdir / "spike_ratio_over_time.png"
    plt.savefig(out, dpi=200)
    plt.close()

def plot_top_layers(by_layer: pd.DataFrame, outdir: Path, topk: int = 12):
    top = by_layer.head(topk)
    idx = np.arange(len(top))
    plt.figure()
    plt.bar(idx, top["spike_ratio"])
    plt.xticks(idx, top["layer"], rotation=60, ha="right")
    plt.ylabel("Average spike ratio")
    plt.title(f"Top {topk} Layers by Average Spike Ratio (across time)")
    plt.tight_layout()
    out = outdir / "top_layers_spike_ratio.png"
    plt.savefig(out, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Plot spike ratio from sdt.log")
    ap.add_argument("--log", type=Path, required=True, help="Path to sdt.log")
    ap.add_argument("--out", type=Path, default=Path("./spike_plots"), help="Output directory")
    ap.add_argument("--topk", type=int, default=12, help="How many top layers to show")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = load_spike_table(args.log)
    # Save detailed CSV
    csv_path = args.out / "spike_ratio_detailed.csv"
    df.to_csv(csv_path, index=False)

    # Per time step: mean +/- std across layers
    by_t = df.groupby("t_index")["spike_ratio"].agg(["mean", "std"]).reset_index()
    by_t.to_csv(args.out / "spike_ratio_by_t.csv", index=False)
    plot_over_time(by_t, args.out)

    # Per layer: average across time steps
    by_layer = df.groupby("layer")["spike_ratio"].mean().sort_values(ascending=False).reset_index()
    by_layer.to_csv(args.out / "spike_ratio_by_layer.csv", index=False)
    plot_top_layers(by_layer, args.out, topk=args.topk)

    print("Saved:")
    print(f"- {csv_path}")
    print(f"- {args.out/'spike_ratio_by_t.csv'}")
    print(f"- {args.out/'spike_ratio_by_layer.csv'}")
    print(f"- {args.out/'spike_ratio_over_time.png'}")
    print(f"- {args.out/'top_layers_spike_ratio.png'}")

if __name__ == "__main__":
    main()
