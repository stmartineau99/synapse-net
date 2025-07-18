import argparse
import os
from glob import glob

import pandas as pd
from common import get_paths, get_file_names, ALL_NAMES


def run_az_evaluation(args):
    from synapse_net.ground_truth.az_evaluation import az_evaluation

    seg_key = f"predictions/az/v{args.version}"

    split_folder = "./models_az_thin"
    results = []
    for dataset in args.datasets:
        print(dataset, ":")
        if args.in_path:
            file_paths = glob(os.path.join(args.in_path, dataset, "*.h5"))
            file_names = [os.path.basename(path) for path in file_paths]
        else:
            file_names = get_file_names(dataset, split_folder, split_names=["test"])
        seg_paths, gt_paths = get_paths(dataset, file_names)
        result = az_evaluation(
            seg_paths, gt_paths, seg_key=seg_key, gt_key="/labels/az_merged_v6",
            criterion=args.criterion, dataset=[dataset] * len(seg_paths), threshold=args.threshold,
        )
        results.append(result)

    results = pd.concat(results)
    output_path = f"/user/muth9/u12095/synapse-net/scripts/cooper/revision/evaluation_results/v{args.version}.xlsx"

    if os.path.exists(output_path):
        # Read existing data
        existing = pd.read_excel(output_path)

        # Ensure consistent column naming and types
        if "tomo_name" in results.columns and "tomo_name" in existing.columns:
            # Drop existing entries with matching "tomo_name"
            existing = existing[~existing["tomo_name"].isin(results["tomo_name"])]

        # Combine: old (filtered) + new
        combined = pd.concat([existing, results], ignore_index=True)
    else:
        combined = results

    # Save back to Excel
    combined.to_excel(output_path, index=False)


def visualize_az_evaluation(args):
    from elf.visualisation.metric_visualization import run_metric_visualization
    from synapse_net.ground_truth.az_evaluation import _postprocess, _crop
    from elf.io import open_file

    seg_key = f"predictions/az/v{args.version}"

    split_folder = "./models_az_thin"
    for dataset in args.datasets:
        file_names = get_file_names(dataset, split_folder, split_names=["test"])
        seg_paths, gt_paths = get_paths(dataset, file_names)

        for seg_path, gt_path in zip(seg_paths, gt_paths):

            with open_file(seg_path, "r") as f:
                seg = f[seg_key][:].squeeze()
            with open_file(gt_path, "r") as f:
                gt = f["/labels/az_merged_v6"][:]

            seg = seg > args.threshold

            seg, gt, bb = _crop(seg, gt, return_bb=True)
            with open_file(gt_path, "r") as f:
                image = f["raw"][bb]

            seg = _postprocess(seg, apply_cc=True, min_component_size=5000, iterations=3)
            gt = _postprocess(gt, apply_cc=True, min_component_size=500)

            run_metric_visualization(image, seg, gt, title=os.path.basename(seg_path), criterion=args.criterion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-v", type=int, required=True)
    parser.add_argument("-c", "--criterion", default="iou")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=ALL_NAMES)
    # Set the threshold to None if the AZ prediction already a segmentation.
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--in_path", "-i", default=None)
    args = parser.parse_args()

    if args.visualize:
        visualize_az_evaluation(args)
    else:
        run_az_evaluation(args)


if __name__ == "__main__":
    main()
