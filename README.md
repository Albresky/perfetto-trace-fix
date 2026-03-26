# perfetto-trace-fix
Fix slice missing in perfetto trace, compatiable with PyTorch.profile trace files.

Post-process a PyTorch Profiler trace for Perfetto compatibility.

Fixes two bugs in traces exported by torch.profiler:

  1. Overlapping slices on the same tid cause Perfetto to hide the later slice.
     Fix: move overlapping slices to a new virtual tid (e.g. "<tid>_hack") until
     they no longer overlap with any existing track.

  2. After a slice is moved to a new tid, flow events ("s"/"f") that were bound
     to that slice still reference the original tid, causing arrows to point at
     the wrong slice.
     Fix: update flow event tids to match their relocated slice.

Usage:
    `python fix_trace.py /abs/path/to/trace.json.gz`