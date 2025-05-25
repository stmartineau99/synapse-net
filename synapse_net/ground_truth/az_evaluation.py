import os
from typing import List

import h5py
import pandas as pd
import numpy as np

from elf.evaluation.matching import _compute_scores, _compute_tps
from elf.evaluation import dice_score
from skimage.measure import label
from tqdm import tqdm


def _postprocess(data, apply_cc, min_component_size):
    if apply_cc:
        data = label(data)
        ids, sizes = np.unique(data, return_counts=True)
        filter_ids = ids[sizes < min_component_size]
        data[np.isin(data, filter_ids)] = 0
    return data


def _single_az_evaluation(seg, gt, apply_cc, min_component_size):
    assert seg.shape == gt.shape, f"{seg.shape}, {gt.shape}"
    seg = _postprocess(seg, apply_cc, min_component_size)
    gt = _postprocess(gt, apply_cc, min_component_size)

    dice = dice_score(seg > 0, gt > 0)

    n_true, n_matched, n_pred, scores = _compute_scores(seg, gt, criterion="iou", ignore_label=0)
    tp = _compute_tps(scores, n_matched, threshold=0.5)
    fp = n_pred - tp
    fn = n_true - tp

    return {"tp": tp, "fp": fp, "fn": fn, "dice": dice}


# TODO further post-processing?
def az_evaluation(
    seg_paths: List[str],
    gt_paths: List[str],
    seg_key: str,
    gt_key: str,
    apply_cc: bool = True,
    min_component_size: int = 100,  # TODO
) -> pd.DataFrame:
    """Evaluate active zone segmentations against ground-truth annotations.

    Args:
        seg_paths: The filepaths to the segmentations, stored as hd5 files.
        gt_paths: The filepaths to the ground-truth annotatons, stored as hdf5 files.
        seg_key: The internal path to the data in the segmentation hdf5 file.
        gt_key: The internal path to the data in the ground-truth hdf5 file.
        apply_cc: Whether to apply connected components before evaluation.
        min_component_size: Minimum component size for filtering the segmentation and annotations before evaluation.

    Returns:
        A data frame with the evaluation results per tomogram.
    """
    assert len(seg_paths) == len(gt_paths)

    results = {
        "tomo_name": [],
        "tp": [],
        "fp": [],
        "fn": [],
        "dice": [],
    }
    for seg_path, gt_path in tqdm(zip(seg_paths, gt_paths), total=len(seg_paths), desc="Run AZ Eval"):
        with h5py.File(seg_path, "r") as f:
            seg = f[seg_key][:]
        with h5py.File(gt_path, "r") as f:
            gt = f[gt_key][:]
        # TODO more post-processing params
        result = _single_az_evaluation(seg, gt, apply_cc, min_component_size)
        results["tomo_name"].append(os.path.basename(seg_path))
        for res in ("tp", "fp", "fn", "dice"):
            results[res].append(result[res])
    return pd.DataFrame(results)
