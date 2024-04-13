"""
Loss functions for recommender models.

The pointwise, BPR, and hinge losses are a good fit for
implicit feedback models trained through negative sampling.
"""

import torch


def pointwise_loss(positive_predictions, negative_predictions, mask=None):
    """
    Logistic loss function.
    """

    positives_loss = (1.0 - torch.sigmoid(positive_predictions))
    negatives_loss = torch.sigmoid(negative_predictions)
    loss = (positives_loss + negatives_loss)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def bpr_loss(positive_predictions, negative_predictions, mask=None):
    """
    Bayesian Personalised Ranking pairwise loss function.
    """

    loss = (1.0 - torch.sigmoid(positive_predictions - negative_predictions))

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def hinge_loss(positive_predictions, negative_predictions, mask=None):
    """
    Hinge pairwise loss function.
    """

    loss = torch.clamp(negative_predictions - positive_predictions + 1.0, 0.0)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def adaptive_hinge_loss(positive_predictions, negative_predictions, mask=None):
    """
    Adaptive hinge pairwise loss function.
    Takes a set of predictions for implicitly negative items, and selects those that are highest,
    thus sampling those negatives that are closes to violating the ranking implicit in the pattern of user interactions.
    """

    highest_negative_predictions, _ = torch.max(negative_predictions, 0)

    return hinge_loss(positive_predictions, highest_negative_predictions.squeeze(), mask=mask)
