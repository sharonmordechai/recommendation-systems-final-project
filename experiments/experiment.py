import os
import time
import numpy as np

from sklearn.model_selection import ParameterSampler
from datasets.movielens import get_movielens_dataset
from evaluations.cross_validation import random_train_test_split
from evaluations.evaluation import mrr_score, rmse_score
from experiments.results import Results
from factorization.implicit import ImplicitFactorizationModel
from model_utils.layers import BloomEmbedding, ScaledEmbedding, HASH_FUNCTIONS
from model_utils.networks import Net
from model_utils.torch_utils import set_seed

CUDA = False
NUM_SAMPLES = 20

# sampling these hyperparameters for the baseline process
LEARNING_RATES = [1e-4, 5 * 1e-4, 1e-3, 1e-2, 5 * 1e-2, 1e-1]
LOSSES = ["bpr", "adaptive_hinge"]
BATCH_SIZE = [32, 64, 128, 256, 512]
EMBEDDING_DIM = [32, 64, 128, 256]
N_ITER = list(range(1, 20))
L2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0]


# hyperparameters for this experiment
COMPRESSION_RATIOS = (np.arange(1, 10) / 10).tolist()


def set_hyperparameters(random_state, num):
    params = {
        "n_iter": N_ITER,
        "batch_size": BATCH_SIZE,
        "l2": L2,
        "learning_rate": LEARNING_RATES,
        "loss": LOSSES,
        "embedding_dim": EMBEDDING_DIM,
    }
    sampler = ParameterSampler(params, n_iter=num, random_state=random_state)

    for params in sampler:
        yield params


def build_factorization_model(hyperparameters, train, random_state):
    h = hyperparameters
    set_seed(42, CUDA)

    if h["compression_ratio"] < 1.0:
        item_embeddings = BloomEmbedding(
            train.num_items,
            h["embedding_dim"],
            compression_ratio=h["compression_ratio"],
            num_hash_functions=4,
            padding_idx=0,
            hash_function=hyperparameters["hash_function"],
        )
        user_embeddings = BloomEmbedding(
            train.num_users,
            h["embedding_dim"],
            compression_ratio=h["compression_ratio"],
            num_hash_functions=4,
            padding_idx=0,
            hash_function=hyperparameters["hash_function"],
        )
    else:
        item_embeddings = ScaledEmbedding(
            train.num_items, h["embedding_dim"], padding_idx=0
        )
        user_embeddings = ScaledEmbedding(
            train.num_users, h["embedding_dim"], padding_idx=0
        )

    network = Net(
        train.num_users,
        train.num_items,
        user_embedding_layer=user_embeddings,
        item_embedding_layer=item_embeddings,
    )
    model = ImplicitFactorizationModel(
        loss=h["loss"],
        n_iter=h["n_iter"],
        batch_size=h["batch_size"],
        learning_rate=h["learning_rate"],
        embedding_dim=h["embedding_dim"],
        l2=h["l2"],
        representation=network,
        use_cuda=CUDA,
        random_state=random_state,
    )

    return model


def evaluate_model(model, train, test, validation):
    start_time = time.perf_counter()
    model.fit(train, verbose=True)
    elapsed = time.perf_counter() - start_time

    print(f"Elapsed {elapsed}")

    test_mrr = mrr_score(model, test)
    val_mrr = mrr_score(model, test.tocsr() + validation.tocsr())
    test_rmse = rmse_score(model, test)
    val_rmse = rmse_score(model, validation)
    return test_mrr, val_mrr, test_rmse, val_rmse, elapsed


def run(experiment_name, hash_function, train, test, validation, random_state):
    print(f"Running experiment {experiment_name}....")
    baseline_path = os.path.join(os.path.dirname(experiment_name), "baseline.txt")
    baseline_results = Results(baseline_path)
    best_result = baseline_results.best()

    if best_result is not None:
        print(f"Best result: {best_result}")

    # Find a good baseline
    for i, hyperparameters in enumerate(set_hyperparameters(random_state, NUM_SAMPLES)):
        hyperparameters["batch_size"] = hyperparameters["batch_size"] * 4
        hyperparameters["compression_ratio"] = 1.0
        hyperparameters["hash_function"] = ""

        if hyperparameters in baseline_results:
            print("Done, skipping...")
            continue

        print(f"==> [{i+1}/{NUM_SAMPLES}] Hyperparameters: {hyperparameters}")
        model = build_factorization_model(hyperparameters, train, random_state)
        test_mrr, val_mrr, test_rmse, val_rmse, elapsed = evaluate_model(
            model, train, test, validation
        )

        print(
            f"==> Test MRR: {test_mrr.mean()}, \
                Validation MRR: {val_mrr.mean()}, \
                Test RMSE: {test_rmse}, \
                Validation RMSE: {val_rmse},\
                Elapsed Time: {elapsed}"
        )
        baseline_results.save(
            hyperparameters,
            test_mrr.mean(),
            val_mrr.mean(),
            test_rmse,
            val_rmse,
            elapsed,
        )

    best_baseline = baseline_results.best_baseline()
    print(f"Best baseline: {best_baseline}")
    print("*" * 200)

    results = Results(experiment_name)
    hyperparameters = best_baseline.copy()
    del hyperparameters["test_mrr"]
    del hyperparameters["validation_mrr"]
    del hyperparameters["test_rmse"]
    del hyperparameters["validation_rmse"]
    del hyperparameters["elapsed"]
    if hyperparameters not in results:
        results.save(
            hyperparameters,
            best_baseline["test_mrr"],
            best_baseline["validation_mrr"],
            np.float32(best_baseline["test_rmse"]),
            np.float32(best_baseline["validation_rmse"]),
            best_baseline["elapsed"],
        )
    # Compute compression results
    for compression_ratio in COMPRESSION_RATIOS:
        hyperparameters["compression_ratio"] = compression_ratio
        hyperparameters["hash_function"] = hash_function

        if hyperparameters in results:
            print("Compression computed")
            continue

        print(hyperparameters)
        model = build_factorization_model(hyperparameters, train, random_state)
        test_mrr, val_mrr, test_rmse, val_rmse, elapsed = evaluate_model(
            model, train, test, validation
        )

        print(
            f"Test MRR: {test_mrr.mean()},\
                Validation MRR: {val_mrr.mean()},\
                Test RMSE: {test_rmse}, \
                Validation RMSE: {val_rmse},\
                Elapsed Time: {elapsed}"
        )
        results.save(
            hyperparameters,
            test_mrr.mean(),
            val_mrr.mean(),
            test_rmse,
            val_rmse,
            elapsed,
        )
        print("*" * 200)

    return results


def run_experiment(variant="100K", save_path="../results"):
    np.random.seed(42)
    random_state = np.random.RandomState(100)
    dataset = get_movielens_dataset(variant)
    test_percentage = 0.2

    train, rest = random_train_test_split(
        dataset, test_percentage=test_percentage, random_state=random_state
    )
    test, validation = random_train_test_split(
        rest, test_percentage=0.5, random_state=random_state
    )

    for hash_function in HASH_FUNCTIONS:
        filename = f"{hash_function}_implicit_movielens_results.txt"
        experiment_name = os.path.join(save_path, filename)
        run(experiment_name, hash_function, train, test, validation, random_state)


if __name__ == "__main__":
    run_experiment()
