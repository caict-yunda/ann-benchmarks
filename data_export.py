import argparse
import csv
import json

from ann_benchmarks.datasets import DATASETS, get_dataset
from ann_benchmarks.plotting.utils import compute_metrics_all_runs
from ann_benchmarks.results import load_all_results


def save_by_csv(dfs, filepath):
    with open(filepath, "w", newline="") as csvfile:
        names = list(dfs[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=names)
        writer.writeheader()
        for res in dfs:
            writer.writerow(res)

def save_by_json(dfs, filepath):
    for row in dfs:
        row["count"] = int(row["count"])

    with open(filepath, "w", newline="") as file:
        file.write(json.dumps(dfs, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Path to the output file", required=True)
    parser.add_argument("--recompute", action="store_true", help="Recompute metrics")
    args = parser.parse_args()

    datasets = DATASETS.keys()
    dfs = []
    for dataset_name in datasets:
        print("Looking at dataset", dataset_name)
        if len(list(load_all_results(dataset_name))) > 0:
            results = load_all_results(dataset_name)
            dataset, _ = get_dataset(dataset_name)
            results = compute_metrics_all_runs(dataset, results, args.recompute)
            for res in results:
                res["dataset"] = dataset_name
                dfs.append(res)
    if len(dfs) > 0:
        if args.output.find(".json") >= 0:
            save_by_json(dfs, args.output)
        else:
            save_by_csv(dfs,  args.output)
