# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.results import Results
from model_utils.layers import HASH_FUNCTIONS

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def like_baseline(baseline, x):
    for key in (
        "n_iter",
        "batch_size",
        "l2",
        "learning_rate",
        "loss",
        "embedding_dim",
    ):
        if x[key] != baseline[key]:
            return False

    return True


def process_results(results, verbose=False):
    baseline = results.best_baseline()
    data = pd.DataFrame([x for x in results if like_baseline(baseline, x)])
    best = (
        data.sort_values("test_mrr", ascending=False)
        .groupby("compression_ratio", as_index=False)
        .first()
    )

    # Normalize per iteration
    best["elapsed"] = best["elapsed"] / best["n_iter"]

    if verbose:
        print(best)

    baseline_mrr = best[best["compression_ratio"] == 1.0]["validation_mrr"].values[0]
    baseline_rmse = best[best["compression_ratio"] == 1.0]["validation_rmse"].values[0]
    baseline_time = best[best["compression_ratio"] == 1.0]["elapsed"].values[0]
    compression_ratio = best["compression_ratio"].values
    mrr = best["validation_mrr"].values / baseline_mrr
    rmse = best["validation_rmse"].values / baseline_rmse
    elapsed = best["elapsed"].values / baseline_time

    return compression_ratio, mrr, rmse, elapsed


def create_compression_plot(results, param, name, save_path):
    processed_results = {}
    plt.figure(figsize=(18, 10))
    label_prompt = {"mrr": "MRR", "rmse": "RMSE", "time": "Time"}
    for result in results:
        (
            compression_ratio,
            processed_results["mrr"],
            processed_results["rmse"],
            processed_results["time"],
        ) = process_results(result, verbose=True)
        plt.plot(
            compression_ratio, processed_results[param], label=result.get_filename()
        )

    plt.ylabel(f"{label_prompt[param]} ratio to baseline")
    plt.xlabel("Compression ratio")
    plt.title(f"Compression ratio vs {label_prompt[param]} ratio")
    plt.gca().invert_xaxis()
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, f"_{name}_{param}_plot.png"))

    plt.close()


def plot_results(model, results, save_path):
    sns.set_style("darkgrid")
    create_compression_plot(results, "mrr", model, save_path)
    create_compression_plot(results, "rmse", model, save_path)
    create_compression_plot(results, "time", model, save_path)


# %%
def plot_experiment_results(save_path="../results"):
    results = []
    for hash_function in HASH_FUNCTIONS:
        filename = f"{hash_function}_implicit_movielens_results.txt"
        filename_path = os.path.join(save_path, filename)
        results.append(Results(filename_path))

    plot_results("MovieLens", results, save_path)


if __name__ == "__main__":
    plot_experiment_results()
