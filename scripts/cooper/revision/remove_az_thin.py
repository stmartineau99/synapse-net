import h5py

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