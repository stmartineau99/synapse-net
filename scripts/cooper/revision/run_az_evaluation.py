import argparse
import os
from glob import glob


def _get_paths(seg_root, gt_root, image_root=None, dataset=None):
    seg_paths = sorted(glob(os.path.join(seg_root, "**/*.h5"), recursive=True))
    if dataset is not None:
        seg_paths = [path for path in seg_paths if os.path.basename(os.path.split(path)[0]).startswith(dataset)]

    gt_paths = []
    for path in seg_paths:
        gt_path = os.path.join(gt_root, os.path.relpath(path, seg_root))
        assert os.path.exists(gt_path), gt_path
        gt_paths.append(gt_path)

    if image_root is None:
        image_paths = [None] * len(seg_paths)
    else:
        image_paths = []
        for path in seg_paths:
            im_path = os.path.join(image_root, os.path.relpath(path, seg_root))
            assert os.path.exists(im_path), im_path
            image_paths.append(im_path)

    return seg_paths, gt_paths, image_paths


def run_az_evaluation(args):
    from synapse_net.ground_truth.az_evaluation import az_evaluation

    seg_paths, gt_paths, _ = _get_paths(args.seg_root, args.gt_root, dataset=args.dataset)
    dataset = [os.path.basename(os.path.split(path)[0]) for path in seg_paths]
    result = az_evaluation(
        seg_paths, gt_paths, seg_key=args.seg_key, gt_key="/labels/az", dataset=dataset, criterion=args.criterion
    )

    if args.output is None:
        output_path = f"./results/{args.seg_key.replace('/', '_')}.xlsx"
    else:
        output_path = f"./results/{args.output}.xlsx"
    result.to_excel(output_path, index=False)


def visualize_az_evaluation(args):
    from elf.visualisation.metric_visualization import run_metric_visualization
    from synapse_net.ground_truth.az_evaluation import _postprocess, _crop
    from elf.io import open_file

    seg_paths, gt_paths, image_paths = _get_paths(args.seg_root, args.gt_root, args.image_root, dataset=args.dataset)
    for seg_path, gt_path, image_path in zip(seg_paths, gt_paths, image_paths):
        image = None if image_path is None else open_file(image_path, "r")["raw"][:]

        with open_file(seg_path, "r") as f:
            seg = f[args.seg_key][:]
        with open_file(gt_path, "r") as f:
            gt = f["/labels/az"][:]

        seg, gt, bb = _crop(seg, gt, return_bb=True)
        if image is not None:
            image = image[bb]

        seg = _postprocess(seg, apply_cc=True, min_component_size=10000, iterations=3)
        gt = _postprocess(gt, apply_cc=True, min_component_size=500)

        run_metric_visualization(image, seg, gt, title=os.path.basename(seg_path), criterion=args.criterion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seg_root", required=True)
    parser.add_argument("-g", "--gt_root", required=True)
    parser.add_argument("--seg_key", required=True)
    parser.add_argument("-i", "--image_root")
    parser.add_argument("-o", "--output")
    parser.add_argument("-c", "--criterion", default="iou")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    if args.visualize:
        visualize_az_evaluation(args)
    else:
        run_az_evaluation(args)


if __name__ == "__main__":
    main()
