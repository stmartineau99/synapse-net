import sys
import os

# Add membrain-seg to Python path
MEMBRAIN_SEG_PATH = "/user/muth9/u12095/membrain-seg/src"
if MEMBRAIN_SEG_PATH not in sys.path:
    sys.path.insert(0, MEMBRAIN_SEG_PATH)

import argparse
import h5py
import pandas as pd
from tqdm import tqdm
import numpy as np

from membrain_seg.segmentation.skeletonize import skeletonization
from membrain_seg.benchmark.metrics import masked_surface_dice


def load_segmentation(file_path, key):
    """Load a dataset from an HDF5 file."""
    with h5py.File(file_path, "r") as f:
        data = f[key][:]
    return data


def evaluate_surface_dice(pred, gt, raw, check):
    """Skeletonize predictions and GT, compute surface dice."""
    gt_skeleton = skeletonization(gt == 1, batch_size=100000)
    pred_skeleton = skeletonization(pred, batch_size=100000)
    mask = gt != 2

    if check:
        import napari
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(gt, name= f"gt")
        v.add_labels(gt_skeleton.astype(np.uint16), name= f"gt_skeleton")
        v.add_labels(pred, name= f"pred")
        v.add_labels(pred_skeleton.astype(np.uint16), name= f"pred_skeleton")
        napari.run()

    surf_dice, confusion_dict = masked_surface_dice(
        pred_skeleton, gt_skeleton, pred, gt, mask
    )
    return surf_dice, confusion_dict


def process_file(pred_path, gt_path, seg_key, gt_key, check):
    """Process a single prediction/GT file pair."""
    try:
        pred = load_segmentation(pred_path, seg_key)
        gt = load_segmentation(gt_path, gt_key)
        raw = load_segmentation(gt_path, "raw")
        surf_dice, confusion = evaluate_surface_dice(pred, gt, raw, check)

        result = {
            "tomo_name": os.path.basename(pred_path),
            "surface_dice": surf_dice,
            **confusion,
        }
        return result

    except Exception as e:
        print(f"Error processing {pred_path}: {e}")
        return None


def collect_results(input_folder, gt_folder, version, check=False):
    """Loop through prediction files and compute metrics."""
    results = []
    seg_key = f"predictions/az/seg_v{version}"
    gt_key = "/labels/az_merged"

    for fname in tqdm(os.listdir(input_folder), desc="Processing segmentations"):
        if not fname.endswith(".h5"):
            continue

        pred_path = os.path.join(input_folder, fname)
        gt_path = os.path.join(gt_folder, fname)

        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found for {fname}")
            continue

        result = process_file(pred_path, gt_path, seg_key, gt_key, check)
        if result:
            results.append(result)

    return results


def save_results(results, output_file):
    """Save results as an Excel file."""
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute surface dice for AZ segmentations.")
    parser.add_argument("--input_folder", "-i", required=True, help="Folder with predicted segmentations (.h5)")
    parser.add_argument("--gt_folder", "-gt", required=True, help="Folder with ground truth segmentations (.h5)")
    parser.add_argument("--version", "-v", required=True, help="Version string used in prediction key")
    parser.add_argument("--check", action="store_true", help="Version string used in prediction key")

    args = parser.parse_args()

    output_file = f"/user/muth9/u12095/synapse-net/scripts/cooper/revision/evaluation_results/v{args.version}_surface_dice.xlsx"
    results = collect_results(args.input_folder, args.gt_folder, args.version, args.check)
    save_results(results, output_file)


if __name__ == "__main__":
    main()