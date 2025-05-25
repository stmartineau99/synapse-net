import argparse
import os
from glob import glob


def _get_paths(seg_root, gt_root, image_root=None):
    seg_paths = sorted(glob(os.path.join(seg_root, "**/*.h5"), recursive=True))
    gt_paths = sorted(glob(os.path.join(gt_root, "**/*.h5"), recursive=True))
    assert len(seg_paths) == len(gt_paths)

    if image_root is None:
        image_paths = [None] * len(seg_paths)
    else:
        image_paths = sorted(glob(os.path.join(image_root, "**/*.mrc"), recursive=True))
        assert len(image_paths) == len(seg_paths)

    return seg_paths, gt_paths, image_paths


# TODO extend this
def run_az_evaluation(args):
    from synapse_net.ground_truth.az_evaluation import az_evaluation

    seg_paths, gt_paths, _ = _get_paths(args.seg_root, args.gt_root)
    result = az_evaluation(seg_paths, gt_paths, seg_key="seg", gt_key="gt")

    print(result)


def visualize_az_evaluation(args):
    from elf.visualisation.metric_visualization import run_metric_visualization
    from synapse_net.ground_truth.az_evaluation import _postprocess
    from elf.io import open_file

    seg_paths, gt_paths, image_paths = _get_paths(args.seg_root, args.gt_root, args.image_root)
    for seg_path, gt_path, image_path in zip(seg_paths, gt_paths, image_paths):
        image = None if image_path is None else open_file(image_path, "r")["data"][:]

        with open_file(seg_path, "r") as f:
            seg = f["seg"][:]
        with open_file(gt_path, "r") as f:
            gt = f["gt"][:]

        seg = _postprocess(seg, apply_cc=True, min_component_size=100)
        gt = _postprocess(gt, apply_cc=True, min_component_size=100)

        run_metric_visualization(image, seg, gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seg_root", required=True)
    parser.add_argument("-g", "--gt_root", required=True)
    parser.add_argument("-i", "--image_root")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    if args.visualize:
        visualize_az_evaluation(args)
    else:
        run_az_evaluation(args)


if __name__ == "__main__":
    main()
