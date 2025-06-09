'''import h5py

files = [
    "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/stem_cropped2_rescaled/36859_H2_SP_02_rec_2Kb1dawbp_crop_crop1.h5",
    "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/stem_cropped2_rescaled/36859_H2_SP_07_rec_2Kb1dawbp_crop_crop1.h5"
]

for file in files:
    with h5py.File(file, "r+") as f:
        # Load the replacement data
        gt = f["labels/az"][:]

        # Delete the existing dataset if it exists
        if "labels/az_thin" in f:
            del f["labels/az_thin"]

        # Recreate the dataset with the new data
        f.create_dataset("labels/az_thin", data=gt)
'''
import h5py
import numpy as np
import os
from glob import glob

folder = "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/stem_cropped/"

# List of file names to process
file_names = [
    "36859_H2_SP_01_rec_2Kb1dawbp_crop_cropped_noAZ.h5",
    "36859_H2_SP_02_rec_2Kb1dawbp_crop_cropped_noAZ.h5",
    "36859_H2_SP_03_rec_2Kb1dawbp_crop_cropped_noAZ.h5",
    "36859_H3_SP_05_rec_2kb1dawbp_crop_cropped_noAZ.h5",
    "36859_H3_SP_07_rec_2kb1dawbp_crop_cropped_noAZ.h5",
    "36859_H3_SP_10_rec_2kb1dawbp_crop_cropped_noAZ.h5"
]

file_paths = glob(os.path.join("/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/endbulb_of_held_cropped", "*.h5"))

for fname in file_paths:
    #file_path = os.path.join(folder, fname)
    
    with h5py.File(fname, "a") as f:
        az_merged = f["/labels/az_merged"][:]
        f.create_dataset("/labels/az_merged_v6", data=az_merged, compression="lzf")

    print(f"Updated file: {fname}")
