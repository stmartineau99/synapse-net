import os
from glob import glob

import h5py
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

ROOT = "/mnt/ceph-hdd/cold_store/projects/nim00007/AZ_data/training_data"
INTER_ROOT = "/mnt/ceph-hdd/cold_store/projects/nim00007/AZ_predictions"
OUTPUT_ROOT = "/mnt/ceph-hdd/cold_store/projects/nim00007/new_AZ_train_data"


def _check_data(files, label_folder, check_thinned):
    for ff in files:
        with h5py.File(ff, "r") as f:
            shape = f["raw"].shape
            az = f["labels/az"][:]
        n_az = az.max()

        if check_thinned:
            label_file = os.path.join(label_folder, os.path.basename(ff))
            with h5py.File(label_file, "r") as f:
                az_thin = f["labels/az_thin2"][:]
            n_az_thin = az_thin.max()
        else:
            n_az_thin = None

        print(os.path.basename(ff), ":", shape, ":", n_az, ":", n_az_thin)


def assort_tem():
    old_name = "01data_withoutInvertedFiles_minusSVseg_corrected"
    new_name = "tem"

    raw_folder = os.path.join(ROOT, old_name)
    label_folder = os.path.join(INTER_ROOT, old_name)
    output_folder = os.path.join(OUTPUT_ROOT, new_name)
    os.makedirs(output_folder, exist_ok=True)

    files = glob(os.path.join(raw_folder, "*.h5"))
    for ff in tqdm(files):
        with h5py.File(ff, "r") as f:
            raw = f["raw"][:]
            az = f["labels/az"][:]

        label_path = os.path.join(label_folder, os.path.basename(ff))
        with h5py.File(label_path, "r") as f:
            az_thin = f["labels/az_thin2"][:]

        z_range1 = np.where(az != 0)[0]
        z_range2 = np.where(az != 0)[0]
        z_range = slice(
            np.min(np.concatenate([z_range1, z_range2])),
            np.max(np.concatenate([z_range1, z_range2])) + 1,
        )
        raw, az, az_thin = raw[z_range], az[z_range], az_thin[z_range]

        out_path = os.path.join(output_folder, os.path.basename(ff))
        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="lzf")
            f.create_dataset("labels/az_thin", data=az_thin, compression="lzf")
            f.create_dataset("labels/az", data=az, compression="lzf")


def assort_chemical_fixation():
    old_name = "12_chemical_fix_cryopreparation_minusSVseg_corrected"
    new_name = "chemical_fixation"

    raw_folder = os.path.join(ROOT, old_name)
    label_folder = os.path.join(INTER_ROOT, old_name)
    output_folder = os.path.join(OUTPUT_ROOT, new_name)
    os.makedirs(output_folder, exist_ok=True)

    label_key = "labels/az_thin2"

    files = glob(os.path.join(raw_folder, "*.h5"))
    for ff in tqdm(files):
        with h5py.File(ff, "r") as f:
            raw = f["raw"][:]
            az = f["labels/az"][:]

        label_path = os.path.join(label_folder, os.path.basename(ff))
        with h5py.File(label_path, "r") as f:
            az_thin = f[label_key][:]

        z_range1 = np.where(az != 0)[0]
        z_range2 = np.where(az != 0)[0]
        z_range = slice(
            np.min(np.concatenate([z_range1, z_range2])),
            np.max(np.concatenate([z_range1, z_range2])) + 1,
        )
        raw, az, az_thin = raw[z_range], az[z_range], az_thin[z_range]

        out_path = os.path.join(output_folder, os.path.basename(ff))
        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="lzf")
            f.create_dataset("labels/az_thin", data=az_thin, compression="lzf")
            f.create_dataset("labels/az", data=az, compression="lzf")


def assort_stem():
    old_names = [
        "04_hoi_stem_examples_fidi_and_sarah_corrected",
        "04_hoi_stem_examples_minusSVseg_cropped_corrected",
        "06_hoi_wt_stem750_fm_minusSVseg_cropped_corrected",
    ]
    new_names = ["stem", "stem_cropped", "stem_cropped"]
    for old_name, new_name in zip(old_names, new_names):
        print(old_name)
        raw_folder = os.path.join(ROOT, f"{old_name}_rescaled_tomograms")
        label_folder = os.path.join(INTER_ROOT, old_name)
        files = glob(os.path.join(raw_folder, "*.h5"))

        # _check_data(files, label_folder, check_thinned=True)
        # continue

        output_folder = os.path.join(OUTPUT_ROOT, new_name)
        os.makedirs(output_folder, exist_ok=True)
        for ff in tqdm(files):
            with h5py.File(ff, "r") as f:
                raw = f["raw"][:]
                az = f["labels/az"][:]

            label_path = os.path.join(label_folder, os.path.basename(ff))
            with h5py.File(label_path, "r") as f:
                az_thin = f["labels/az_thin2"][:]
            az_thin = resize(az_thin, az.shape, order=0, anti_aliasing=False, preserve_range=True).astype(az_thin.dtype)
            assert az_thin.shape == az.shape

            out_path = os.path.join(output_folder, os.path.basename(ff))
            with h5py.File(out_path, "a") as f:
                f.create_dataset("raw", data=raw, compression="lzf")
                f.create_dataset("labels/az_thin", data=az_thin, compression="lzf")
                f.create_dataset("labels/az", data=az, compression="lzf")


def assort_wichmann():
    old_name = "wichmann_withAZ_rescaled_tomograms"
    new_name = "endbulb_of_held"

    raw_folder = os.path.join(ROOT, old_name)
    output_folder = os.path.join(OUTPUT_ROOT, new_name)
    os.makedirs(output_folder, exist_ok=True)

    files = glob(os.path.join(raw_folder, "*.h5"))

    output_folder = os.path.join(OUTPUT_ROOT, new_name)
    os.makedirs(output_folder, exist_ok=True)
    for ff in tqdm(files):
        with h5py.File(ff, "r") as f:
            raw = f["raw"][:]
            az = f["labels/az"][:]

        output_file = os.path.join(output_folder, os.path.basename(ff))
        with h5py.File(output_file, "a") as f:
            f.create_dataset("raw", data=raw, compression="lzf")
            f.create_dataset("labels/az", data=az, compression="lzf")
            f.create_dataset("labels/az_thin", data=az, compression="lzf")


def crop_wichmann():
    input_name = "endbulb_of_held"
    output_name = "endbulb_of_held_cropped"

    input_folder = os.path.join(OUTPUT_ROOT, input_name)
    output_folder = os.path.join(OUTPUT_ROOT, output_name)
    os.makedirs(output_folder, exist_ok=True)
    files = glob(os.path.join(input_folder, "*.h5"))

    min_shape = (32, 512, 512)

    for ff in tqdm(files):
        with h5py.File(ff, "r") as f:
            az = f["labels/az"][:]
            bb = np.where(az != 0)
            bb = tuple(slice(int(b.min()), int(b.max()) + 1) for b in bb)
            pad_width = [max(sh - (b.stop - b.start), 0) // 2 for b, sh in zip(bb, min_shape)]
            bb = tuple(
                slice(max(b.start - pw, 0), min(b.stop + pw, sh)) for b, pw, sh in zip(bb, pad_width, az.shape)
            )
            az = az[bb]
            raw = f["raw"][bb]

        # import napari
        # v = napari.Viewer()
        # v.add_image(raw)
        # v.add_labels(az)
        # v.add_labels(az_thin)
        # napari.run()

        output_path = os.path.join(output_folder, os.path.basename(ff).replace(".h5", "_cropped.h5"))
        with h5py.File(output_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="lzf")
            f.create_dataset("labels/az", data=az, compression="lzf")
            f.create_dataset("labels/az_thin", data=az, compression="lzf")


def main():
    # assort_tem()
    # assort_chemical_fixation()

    # assort_stem()

    # assort_wichmann()
    crop_wichmann()


if __name__ == "__main__":
    main()
