import numpy as np
import scipy.stats as st


FLOAT_MAX = np.finfo(np.float32).max


def mrr_score(model, test, train=None):
    """
    Compute mean reciprocal rank (MRR) scores.
    One score is given for every user with interactions in the test set,
    representing the mean reciprocal rank of all their test items.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    mrrs = []

    for user_id, row in enumerate(test):
        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            predictions[train[user_id].indices] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()
        mrrs.append(mrr)

    return np.array(mrrs)


def _get_precision_recall(predictions, targets, k):
    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))

    return float(num_hit) / len(predictions), float(num_hit) / len(targets)


def precision_recall_score(model, test, train=None, k=10):
    """
    Compute Precision@k and Recall@k scores.
    One score is given for every user with interactions in the test set,
    representing the Precision@k and Recall@k of all their test items.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []

    for user_id, row in enumerate(test):
        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX

        predictions = predictions.argsort()
        targets = row.indices
        user_precision, user_recall = zip(
            *[_get_precision_recall(predictions, targets, x) for x in k]
        )

        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()

    return precision, recall


def rmse_score(model, test):
    """
    Compute RMSE score for test interactions.
    """

    predictions = np.clip(model.predict(test.user_ids, test.item_ids), 0, 1)
    return np.sqrt(((test.ratings - predictions) ** 2).mean())
