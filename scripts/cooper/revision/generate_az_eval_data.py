from synapse_net.sample_data import get_sample_data
from synapse_net.inference import run_segmentation, get_model
from elf.io import open_file


sample_data = get_sample_data("tem_tomo")
tomo = open_file(sample_data, "r")["data"][:]

model = get_model("active_zone")
seg = run_segmentation(tomo, model, "active_zone")

with open_file("./pred.h5", "a") as f:
    f.create_dataset("pred", data=seg, compression="gzip")
