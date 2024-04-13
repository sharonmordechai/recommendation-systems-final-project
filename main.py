# %%
import argparse
from experiments.experiment import run_experiment
from experiments.plot import plot_experiment_results

# %%
def main(save_path):
    if save_path is None:
        save_path = "./results"
    run_experiment(save_path=save_path)
    plot_experiment_results(save_path)


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create analysis plots")
    parser.add_argument(
        "-o", "--output", type=str, help="the output directory for model results"
    )
    args = parser.parse_args()
    main(args.output)
