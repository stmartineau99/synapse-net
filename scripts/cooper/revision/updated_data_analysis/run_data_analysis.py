import argparse
import os

from analysis_segmentations import run_predictions
from data_analysis import calc_AZ_SV_distance, calc_SV_diameters, combine_lists, sort_by_distances
from store_results import run_store_results

def run_data_analysis(input_path, output_path, store, resolution, analysis_output):
    print("Starting SV, compartment, and AZ predictions")
    SV_seg, az_seg = run_predictions(input_path, output_path, store)

    print("Performing automatic data analysis")
    print("Calculating per SV distance to AZ")
    dist_list = calc_AZ_SV_distance(SV_seg, az_seg, resolution)

    print("Calculating per SV diameters")
    diam_list = calc_SV_diameters(SV_seg, resolution)

    print("Combining lists")
    combined_list = combine_lists(dist_list, diam_list)

    print("Sorting the combined list by distances")
    sorted_list = sort_by_distances(combined_list)

    print(f"Storing lists under {analysis_output}")
    run_store_results(input_path, analysis_output, sorted_list)


def main():
    parser = argparse.ArgumentParser(description="Run data analysis on HDF5 data.")
    parser.add_argument(
        "--input_path", "-i", type=str, required=True,
        help="Path to an HDF5 file or directory of files."
    )
    parser.add_argument(
        "--analysis_output", "-s", type=str, default = "./analysis_results/",
        help="Path to the folder where the analysis results get saved."
    )
    parser.add_argument(
        "--output_folder", "-o", type=str, default=None,
        help="Optional output folder for storing results."
    )
    parser.add_argument(
        "--store", action="store_true",
        help="Store predictions in output files."
    )
    parser.add_argument(
        "--resolution", type=float, nargs=3, default=(1.554, 1.554, 1.554),
        help="Resolution of input image."
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_folder = args.output_folder
    store = args.store
    resolution = args.resolution
    analysis_output = args.analysis_output

    if os.path.isfile(input_path):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_folder, filename) if output_folder else None
        run_data_analysis(input_path, output_path, store, resolution, analysis_output)

    elif os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.endswith(".h5"):
                full_input_path = os.path.join(input_path, file)
                output_path = os.path.join(output_folder, file) if output_folder else None
                run_data_analysis(full_input_path, output_path, store, resolution, analysis_output)

    else:
        raise ValueError(f"Invalid input path: {input_path}")

    print("Finished!")


if __name__ == "__main__":
    main()