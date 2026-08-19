"""Matched A/B benchmark across model variants.

Runs the same clips through each loaded variant in turn and prints one table.
This is the artifact for the open hardware/metric question: rather than asking
which axis the digital twin grades compression on and waiting for an answer, it
reports size, latency, CPU, and memory together so the decision can be made from
data.

    python benchmark.py --clips "/path/to/your/clips" \
                        --variants fp16_pruned493 fp32_pruned493 \
                        --limit 2000 --out bench_pi4.json

Run it ON the target device with the service running locally. Numbers taken on a
dev laptop do not transfer to a Pi 4, and mixing the two is worse than having
neither.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def run_variant(client: httpx.Client, clips: list[str], variant: str,
                limit: int | None, min_conf: float, detections: bool) -> dict:
    r = client.post("/jobs", json={
        "paths": clips,
        "variant": variant,
        "limit": limit,
        "min_confidence": min_conf,
        "collect_detections": detections,
    })
    r.raise_for_status()
    job_id = r.json()["job_id"]

    last = -1
    while True:
        state = client.get(f"/jobs/{job_id}").json()
        if state["completed"] != last:
            last = state["completed"]
            print(f"\r  {variant}: {last}/{state['total']}", end="", flush=True)
        if state["status"] in ("done", "error", "cancelled"):
            print()
            break
        time.sleep(0.5)

    if state["status"] != "done":
        raise RuntimeError(f"{variant}: {state['status']} — {state.get('error')}")
    return state["metrics"]


def table(rows: list[dict]) -> str:
    cols = [
        ("variant", "variant", "{}"),
        ("size MB", "model_file_mb", "{:.2f}"),
        ("windows", "windows", "{}"),
        ("wall s", "wall_seconds", "{:.1f}"),
        ("win/s", "windows_per_second", "{:.2f}"),
        ("p50 ms", "inference_ms_p50", "{:.2f}"),
        ("p95 ms", "inference_ms_p95", "{:.2f}"),
        ("CPU s", "process_cpu_seconds", "{:.1f}"),
        ("peak RSS MB", "peak_rss_mb", "{}"),
    ]
    widths = [max(len(h), *(len(f.format(r[k]) if r.get(k) is not None else "-")
                            for r in rows)) for h, k, f in cols]
    out = [" | ".join(h.ljust(w) for (h, _, _), w in zip(cols, widths))]
    out.append("-+-".join("-" * w for w in widths))
    for r in rows:
        cells = []
        for (_, k, f), w in zip(cols, widths):
            v = r.get(k)
            cells.append(("-" if v is None else f.format(v)).ljust(w))
        out.append(" | ".join(cells))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-confidence", type=float, default=0.25)
    ap.add_argument("--detections", action="store_true",
                    help="Collect detections too. Off by default: pure latency runs "
                         "should not pay allocation cost for results nobody reads.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    with httpx.Client(base_url=args.host, timeout=None) as client:
        health = client.get("/health").json()
        loaded = set(health["variants_loaded"])
        missing = set(args.variants) - loaded
        if missing:
            print(f"Not loaded: {sorted(missing)}. Available: {sorted(loaded)}", file=sys.stderr)
            return 1
        print(f"onnxruntime {health['onnxruntime_version']} | "
              f"providers {health['providers']}\n")

        rows = [run_variant(client, args.clips, v, args.limit,
                            args.min_confidence, args.detections)
                for v in args.variants]

    print("\n" + table(rows))

    if len(rows) > 1:
        base, *rest = rows
        print(f"\nRelative to {base['variant']}:")
        for r in rest:
            ds = r["model_file_mb"] / base["model_file_mb"]
            dl = r["inference_ms_p50"] / base["inference_ms_p50"] if base["inference_ms_p50"] else float("nan")
            print(f"  {r['variant']}: size x{ds:.2f}, p50 latency x{dl:.2f}")
        print("\nNote: on ORT CPU, FP16 weights are upcast to FP32 for compute "
              "(18 paired cast nodes in the optimized graph), so a flat latency "
              "result here is expected and is not evidence about FP16-native hardware.")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
