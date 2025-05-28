python run_az_evaluation.py \
    -s /mnt/ceph-hdd/cold_store/projects/nim00007/AZ_data/segmentations \
    -g /mnt/ceph-hdd/cold_store/projects/nim00007/AZ_data/training_data \
    --seg_key /AZ/segment_from_AZmodel_TEM_STEM_ChemFix_v1 \
    --criterion iop \
    -o v1
    # --dataset 01 \
    # --seg_key AZ/segment_from_AZmodel_TEM_STEM_ChemFix_wichmann_v2 \
