#!/usr/bin/env python3
import argparse, re, json, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

TOP1_RE = re.compile(r"top-1:\s*([0-9.]+)")
TOP5_RE = re.compile(r"top-5:\s*([0-9.]+)")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def extract_last_value(pattern: re.Pattern, text: str) -> float:
    m = None
    for m in pattern.finditer(text):
        pass
    if not m:
        raise ValueError(f"Could not find value for {pattern.pattern}")
    return float(m.group(1))

def extract_json_block(label: str, text: str) -> Dict[str, Any]:
    """
    Finds the JSON block that comes after a line that equals '<label>:'.
    We then read brace-balanced JSON until it closes.
    """
    lines = text.splitlines()
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().endswith(f"{label}:"):
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(f"Could not find label '{label}:'")

    # accumulate brace-balanced JSON lines
    buf = []
    brace = 0
    started = False
    for ln in lines[start_idx:]:
        # skip empty lines until we hit the opening brace
        if not started:
            if ln.strip().startswith("{"):
                started = True
            else:
                continue
        buf.append(ln)
        brace += ln.count("{") - ln.count("}")
        if started and brace == 0:
            break

    js = "\n".join(buf).strip()
    return json.loads(js)

def fold_sum(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """
    Recursively sum nested dicts of floats.
    """
    for k, v in src.items():
        if isinstance(v, dict):
            if k not in dst: dst[k] = {}
            fold_sum(dst[k], v)
        else:
            dst[k] = dst.get(k, 0.0) + float(v)

def fold_div(dst: Dict[str, Any], denom: float) -> None:
    """
    Recursively divide nested dicts by denom (to get average).
    """
    for k, v in dst.items():
        if isinstance(v, dict):
            fold_div(v, denom)
        else:
            dst[k] = float(v) / denom

def parse_one_log(path: Path) -> Tuple[float, float, Dict[str, Any], Dict[str, Any]]:
    text = read_text(path)
    top1 = extract_last_value(TOP1_RE, text)
    top5 = extract_last_value(TOP5_RE, text)
    non_zero = extract_json_block("non_zero", text)
    firing_rate = extract_json_block("firing_rate", text)
    return top1, top5, non_zero, firing_rate

def main():
    ap = argparse.ArgumentParser(description="Aggregate SpikeDrivenTransformer test logs")
    ap.add_argument("logs", nargs="+",
                    help="List of sdt.log files or a glob (e.g. seed*/sdt.log)")
    ap.add_argument("--outdir", default="./agg_out", help="Where to write summaries")
    args = ap.parse_args()

    # Expand globs to files
    paths: List[Path] = []
    for p in args.logs:
        matched = list(Path().glob(p)) if any(c in p for c in "*?[]") else [Path(p)]
        for m in matched:
            if m.is_dir():
                cand = m / "sdt.log"
                if cand.exists(): paths.append(cand)
            else:
                paths.append(m)

    paths = [p for p in paths if p.name.endswith("sdt.log")]
    if not paths:
        print("No sdt.log files found from given inputs.", file=sys.stderr)
        sys.exit(2)

    paths = sorted(set(paths))
    print(f"Found {len(paths)} logs:")
    for p in paths: print(" -", p)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect & aggregate
    per_run_rows = []
    nz_sum: Dict[str, Any] = {}
    fr_sum: Dict[str, Any] = {}
    t1_sum = 0.0
    t5_sum = 0.0

    for p in paths:
        top1, top5, nz, fr = parse_one_log(p)
        per_run_rows.append({"log": str(p), "top1": top1, "top5": top5})
        t1_sum += top1
        t5_sum += top5
        fold_sum(nz_sum, nz)
        fold_sum(fr_sum, fr)

    n = float(len(paths))
    top1_avg = t1_sum / n
    top5_avg = t5_sum / n
    fold_div(nz_sum, n)
    fold_div(fr_sum, n)

    # Write per-run CSV
    csv_path = outdir / "per_run.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("log,top1,top5\n")
        for r in per_run_rows:
            f.write(f"{r['log']},{r['top1']:.4f},{r['top5']:.4f}\n")

    # Write neat JSON summary
    summary = {
        "num_runs": int(n),
        "mean_top1": round(top1_avg, 4),
        "mean_top5": round(top5_avg, 4),
        "per_run": per_run_rows,
        "non_zero_mean": nz_sum,
        "firing_rate_mean": fr_sum,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Also a tiny TXT for quick viewing
    (outdir / "summary.txt").write_text(
        f"Runs: {int(n)}\nMean top-1: {top1_avg:.4f}\nMean top-5: {top5_avg:.4f}\n"
    )

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", outdir / "summary.json")
    print(" -", outdir / "summary.txt")

if __name__ == "__main__":
    main()
