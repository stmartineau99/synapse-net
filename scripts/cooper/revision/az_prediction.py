import argparse
import os

import h5py
from synapse_net.inference.active_zone import segment_active_zone
from torch_em.util import load_model
from tqdm import tqdm

from common import get_file_names, get_split_folder, ALL_NAMES, INPUT_ROOT, OUTPUT_ROOT


def run_prediction(model, name, split_folder, version, split_names):
    file_names = get_file_names(name, split_folder, split_names=split_names)

    output_folder = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(output_folder, exist_ok=True)
    output_key = f"predictions/az/v{version}"

    for fname in tqdm(file_names):
        output_path = os.path.join(output_folder, fname)

        if os.path.exists(output_path):
            with h5py.File(output_path, "r") as f:
                if output_key in f:
                    continue

        input_path = os.path.join(INPUT_ROOT, name, fname)
        with h5py.File(input_path, "r") as f:
            raw = f["raw"][:]

        _, pred = segment_active_zone(raw, model=model, verbose=False, return_predictions=True)
        with h5py.File(output_path, "a") as f:
            f.create_dataset(output_key, data=pred, compression="lzf")


def get_model(version):
    assert version in (3, 4, 5)
    split_folder = get_split_folder(version)
    if version == 3:
        model_path = os.path.join(split_folder, "checkpoints", "3D-AZ-model-TEM_STEM_ChemFix_wichmann-v3")
    else:
        model_path = os.path.join(split_folder, "checkpoints", f"v{version}")
    model = load_model(model_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-v", type=int)
    parser.add_argument("--names", nargs="+", default=ALL_NAMES)
    parser.add_argument("--splits", nargs="+", default=["test"])
    args = parser.parse_args()

    model = get_model(args.version)
    split_folder = get_split_folder(args.version)
    for name in args.names:
        run_prediction(model, name, split_folder, args.version, args.splits)


if __name__ == "__main__":
    main()
