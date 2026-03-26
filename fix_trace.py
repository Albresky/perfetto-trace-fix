"""
fix_trace.py — Post-process a PyTorch Profiler trace for Perfetto compatibility.

Fixes two bugs in traces exported by torch.profiler:

  1. Overlapping slices on the same tid cause Perfetto to hide the later slice.
     Fix: move overlapping slices to a new virtual tid (e.g. "<tid>_hack") until
     they no longer overlap with any existing track.

  2. After a slice is moved to a new tid, flow events ("s"/"f") that were bound
     to that slice still reference the original tid, causing arrows to point at
     the wrong slice.
     Fix: update flow event tids to match their relocated slice.

Usage:
    python fix_trace.py /abs/path/to/trace.json.gz
"""

import gzip
from collections import defaultdict
from pathlib import Path

import orjson
import typer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(path: str) -> None:
    path_input = Path(path)
    path_output = path_input.parent / f"perfetto-compatible-{path_input.name}"
    print(f"input : {path_input}")
    print(f"output: {path_output}")

    with gzip.open(path_input, "rt", encoding="utf-8") as f:
        trace = orjson.loads(f.read())

    events = trace.get("traceEvents", [])
    fixed_events = _fix_events(events)

    output = {k: v for k, v in trace.items() if k != "traceEvents"}
    output["traceEvents"] = fixed_events

    with gzip.open(path_output, "wb") as f:
        f.write(orjson.dumps(output))

    print("done.")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _fix_events(events: list) -> list:
    """Fix overlapping slices and misbound flow arrows in-place."""
    print(f"total events: {len(events)}")

    relocated = _fix_overlapping_slices(events)
    _fix_flow_events(events, relocated)

    return events


def _fix_overlapping_slices(events: list) -> dict:
    """
    Move CUDA kernel slices that overlap on the same tid to a new virtual tid.

    Returns a dict mapping (pid, original_tid, ts) -> new_tid for every slice
    that was relocated, so flow events can be updated accordingly.
    """
    # Tracks the latest end-time seen for each (pid, tid).
    track_end: dict[tuple, float] = defaultdict(lambda: -1)
    relocated: dict[tuple, object] = {}
    moved = 0

    for e in events:
        if e.get("ph") != "X" or not _is_cuda_kernel(e):
            continue

        original_tid = e["tid"]

        # Bump to a new virtual tid until there is no overlap.
        while e["ts"] < track_end[(e["pid"], e["tid"])]:
            e["tid"] = str(e["tid"]) + "_hack"

        if e["tid"] != original_tid:
            relocated[(e["pid"], original_tid, e["ts"])] = e["tid"]
            moved += 1

        track_end[(e["pid"], e["tid"])] = e["ts"] + e["dur"]

    print(f"relocated {moved} overlapping slices")
    return relocated


def _fix_flow_events(events: list, relocated: dict) -> None:
    """
    Update flow events whose bind-point (pid, tid, ts) was on a relocated slice.

    Flow events use (pid, tid, ts) to anchor themselves to a slice.  After we
    rename a slice's tid we must update the matching flow event so the arrow
    continues to point at (and from) the correct slice.
    """
    fixed = 0

    for e in events:
        if e.get("ph") not in ("s", "f", "t"):
            continue
        key = (e["pid"], e["tid"], e["ts"])
        if key in relocated:
            e["tid"] = relocated[key]
            fixed += 1

    print(f"updated {fixed} flow events")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_cuda_kernel(e: dict) -> bool:
    """Return True for CUDA kernel slices (identified by the 'registers per thread' arg)."""
    return "registers per thread" in e.get("args", {})


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    typer.run(main)
