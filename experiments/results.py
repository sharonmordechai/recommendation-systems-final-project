import os
import hashlib
import json


class Results:
    def __init__(self, filename):
        self._filename = filename
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        f = open(self._filename, "a+")
        f.close()

    def save(
        self, hyperparams, test_mrr, validation_mrr, test_rmse, validation_rmse, elapsed
    ):
        result = hyperparams.copy()
        result.update(
            {
                "test_mrr": test_mrr,
                "validation_mrr": validation_mrr,
                "test_rmse": test_rmse.item(),
                "validation_rmse": validation_rmse.item(),
                "elapsed": elapsed,
                "hash": self._hash(hyperparams),
            }
        )

        with open(self._filename, "a+") as out:
            out.write(json.dumps(result) + "\n")

    def best_baseline(self):
        baseline_results = [
            x
            for x in self
            if x["compression_ratio"] == 1.0 and x["embedding_dim"] >= 32
        ]
        results = sorted(baseline_results, key=lambda x: -x["test_mrr"])
        return results[0] if results else None

    def best(self):
        results = sorted([x for x in self], key=lambda x: -x["test_mrr"])
        return results[0] if results else None

    def get_filename(self):
        return os.path.basename(self._filename).split("_")[0]

    def __getitem__(self, hyperparams):
        params_hash = self._hash(hyperparams)

        with open(self._filename, "r+") as fle:
            for line in fle:
                datum = json.loads(line)

                if datum["hash"] == params_hash:
                    del datum["hash"]
                    return datum

        raise KeyError

    def __contains__(self, x):
        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):
        with open(self._filename, "r+") as fle:
            for line in fle:
                datum = json.loads(line)
                del datum["hash"]
                yield datum

    @staticmethod
    def _hash(x):
        return hashlib.md5(json.dumps(x, sort_keys=True).encode("utf-8")).hexdigest()

