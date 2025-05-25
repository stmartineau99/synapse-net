from synapse_net.sample_data import get_sample_data
from elf.io import open_file


sample_data = get_sample_data("tem_tomo")
tomo = open_file(sample_data, "r")["data"][:]


def run_prediction():
    from synapse_net.inference import run_segmentation, get_model

    model = get_model("active_zone")
    seg = run_segmentation(tomo, model, "active_zone")

    with open_file("./pred.h5", "a") as f:
        f.create_dataset("pred", data=seg, compression="gzip")


def check_prediction():
    import napari

    with open_file("./pred.h5", "r") as f:
        pred = f["pred"][:]

    v = napari.Viewer()
    v.add_image(tomo)
    v.add_labels(pred)
    napari.run()


check_prediction()
