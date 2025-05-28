import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("result_path")
args = parser.parse_args()

results = pd.read_excel(args.result_path)


def summarize_results(res):
    print("Dice-Score:", res["dice"].mean(), "+-", res["dice"].std())
    tp, fp, fn = float(res["tp"].sum()), float(res["fp"].sum()), float(res["fn"].sum())
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * tp / (2 * tp + fn + fp)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1_score)


# # Compute the results for Chemical Fixation.
results_chem_fix = results[results.dataset.str.startswith("12")]
if results_chem_fix.size > 0:
    print("Chemical Fixation Results:")
    summarize_results(results_chem_fix)
#
# # Compute the results for STEM (=04).
results_stem = results[results.dataset.str.startswith(("04", "06"))]
if results_stem.size > 0:
    print()
    print("STEM Results:")
    summarize_results(results_stem)
#
# # Compute the results for TEM (=01).
results_tem = results[results.dataset.str.startswith("01")]
if results_tem.size > 0:
    print()
    print("TEM Results:")
    summarize_results(results_tem)

#
# Compute the results for Wichmann.
results_wichmann = results[results.dataset.str.startswith("wichmann")]
if results_wichmann.size > 0:
    print()
    print("Wichmann Results:")
    summarize_results(results_wichmann)
