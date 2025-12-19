import json

# import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def return_elapsed_seconds(stats):
    start_string = stats["metadata"]["start_time"]
    end_string = stats["metadata"]["end_time"]
    fmt = "%Y-%m-%d %H:%M:%S.%f%z"
    delta = datetime.strptime(end_string, fmt) - datetime.strptime(start_string, fmt)
    return delta.total_seconds()


if __name__ == "__main__":
    # outdir = Path(os.environ.get('MEMRAY_RESULTS_DIR', Path(__file__).parents[1]))
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1]
    trace_paths = (outdir / ".memray-trackings").glob("*stats*.json")
    results = []
    for trace_path in trace_paths:
        with open(trace_path, "r") as f:
            stats = json.load(f)
        data, format, _, method, test, nthreads, version = trace_path.stem.split("-")[3:10]
        pmu = stats["metadata"]["peak_memory"] / 1000 / 1000 / 1000  # bytes to GB
        results.append(
            {
                "method": method,
                "test": test,
                "data": data,
                "format": format,
                # "norm": norm,
                "nthreads": int(nthreads.split("=")[1]),
                "peak_memory_GB": pmu,
                "elapsed_seconds": return_elapsed_seconds(stats),
                "version": int(version.rstrip(".bin")),
                "file": trace_path.name,
            }
        )
    results = pd.DataFrame(results)
    # Grab the latest version only
    results = (
        results.groupby(results.columns[:-1].tolist(), as_index=False)
        .apply(lambda df: df.sort_values("version").iloc[-1])
        .reset_index(drop=True)
    )
    # results.to_csv(Path(__file__).parents[1] / 'footprint-comparison.csv', index=False)
    comp = results.pivot_table(index=["test", "data", "format"], columns="method", values="peak_memory_GB")
    print(comp)
