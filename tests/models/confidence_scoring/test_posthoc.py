import numpy as np
import pytest
import torch

from segmentation_failures.models.pixel_confidence.posthoc import (
    compute_confidence_map,
    compute_mean_prediction,
)
from segmentation_failures.models.pixel_confidence.scores import (
    MaximumSoftmaxScore,
    PredictiveEntropyScore,
    compute_entropy,
)


def test_msr():
    msr_score_p = MaximumSoftmaxScore()

    NUM_CLASSES = 4
    NUM_BATCH = 2
    IMG_SIZE = (3, 3)

    # random tensor
    logits_tensor = torch.randn(NUM_BATCH, NUM_CLASSES, *IMG_SIZE)
    softmax_tensor = torch.softmax(logits_tensor, axis=1)
    true_value = torch.amax(softmax_tensor, dim=1)
    # with probs
    test_value = msr_score_p(softmax_tensor)
    assert torch.all(true_value == test_value), f"Expected {true_value},\n got {test_value}"


def test_entropy():
    NUM_CLASSES = 4
    NUM_BATCH = 2
    IMG_SIZE = (3, 3)

    # entropy of uniform distribution == log(num_classes)
    softmax_tensor = torch.ones(NUM_BATCH, NUM_CLASSES, *IMG_SIZE)
    softmax_tensor = softmax_tensor / softmax_tensor.sum(axis=1, keepdim=True)
    true_value = torch.log(NUM_CLASSES * torch.ones(NUM_BATCH, *IMG_SIZE))
    test_value = compute_entropy(softmax_tensor)
    assert torch.all(true_value == test_value), f"Expected {true_value},\n got {test_value}"

    # entropy of delta distribution == 0
    softmax_tensor = torch.zeros(NUM_BATCH * int(np.prod(IMG_SIZE)), NUM_CLASSES)
    softmax_tensor[:, 0] = 1
    for i, s_row in enumerate(softmax_tensor):
        softmax_tensor[i] = s_row[torch.randperm(len(s_row))]
    softmax_tensor = softmax_tensor.reshape(NUM_BATCH, *IMG_SIZE, NUM_CLASSES).permute(
        dims=(0, 3, 1, 2)
    )
    assert torch.all(softmax_tensor.sum(dim=1) == 1), softmax_tensor.sum(dim=1)

    true_value = torch.zeros(NUM_BATCH, *IMG_SIZE)
    test_value = compute_entropy(softmax_tensor)
    assert torch.all(true_value == test_value), f"Expected {true_value},\n got {test_value}"

    from scipy.stats import entropy

    # random tensor
    softmax_tensor = torch.rand(NUM_BATCH, NUM_CLASSES, *IMG_SIZE)
    softmax_tensor = softmax_tensor / softmax_tensor.sum(axis=1, keepdim=True)
    true_value = torch.tensor(entropy(softmax_tensor.numpy(), axis=1))
    test_value = compute_entropy(softmax_tensor)
    assert torch.all(true_value.isclose(test_value)), f"Expected {true_value},\n got {test_value}"


def test_two_class_msr_vs_entropy():
    msr_score = MaximumSoftmaxScore()
    entropy_score = PredictiveEntropyScore()

    NUM_BATCH = 10
    IMG_SIZE = (64, 64)
    DTYPE = torch.float64
    TOLERANCE = 0

    torch.manual_seed(587465216)
    input_tensor1 = torch.rand(NUM_BATCH, 1, *IMG_SIZE, dtype=DTYPE)
    input_tensor2 = torch.rand(NUM_BATCH, 1, *IMG_SIZE, dtype=DTYPE)
    input_tensor1 = torch.cat([input_tensor1, 1 - input_tensor1], dim=1)
    input_tensor2 = torch.cat([input_tensor2, 1 - input_tensor2], dim=1)

    msr_confid1 = msr_score(input_tensor1)
    msr_confid2 = msr_score(input_tensor2)
    ent_confid1 = entropy_score(input_tensor1)
    ent_confid2 = entropy_score(input_tensor2)

    # check that the ranking of two cases is the same
    # same_ranking = torch.logical_or(
    #     torch.logical_and(msr_confid1 > msr_confid2, ent_confid1 > ent_confid2),
    #     torch.logical_or(
    #         torch.logical_and(msr_confid1 == msr_confid2, ent_confid1 == ent_confid2),
    #         torch.logical_and(msr_confid1 < msr_confid2, ent_confid1 < ent_confid2)
    #     )
    # )
    swapped_ranks = torch.logical_or(
        torch.logical_and(msr_confid1 > msr_confid2, ent_confid1 < ent_confid2),
        torch.logical_and(msr_confid1 < msr_confid2, ent_confid1 > ent_confid2),
    )
    same_msr_different_ent = torch.logical_and(
        msr_confid1 == msr_confid2, ent_confid1 != ent_confid2
    )
    same_ent_different_msr = torch.logical_and(
        msr_confid1 != msr_confid2, ent_confid1 == ent_confid2
    )

    swapped_ranks = swapped_ranks.float().mean().item()
    same_msr_different_ent = same_msr_different_ent.float().mean().item()
    same_ent_different_msr = same_ent_different_msr.float().mean().item()
    assert swapped_ranks <= TOLERANCE
    assert same_msr_different_ent <= TOLERANCE
    assert same_ent_different_msr <= TOLERANCE


def test_maxsoftmax_mc_samples():
    msr_score_p = MaximumSoftmaxScore()

    NUM_SAMPLES = 10
    NUM_CLASSES = 4
    NUM_BATCH = 2
    IMG_SIZE = (3, 3)

    # random tensor
    logits_tensor = torch.randn(NUM_SAMPLES, NUM_BATCH, NUM_CLASSES, *IMG_SIZE)
    softmax_tensor = torch.softmax(logits_tensor, axis=2)
    mean_softmax = softmax_tensor.mean(dim=0)
    true_value = torch.amax(mean_softmax, dim=1)
    # with probs
    test_value = msr_score_p(softmax_tensor, mc_samples_dim=0)
    assert torch.all(true_value == test_value), f"Expected {true_value},\n got {test_value}"


@pytest.mark.parametrize("mc_samples", [1, 10])
@pytest.mark.parametrize("overlapping_classes", [False, True])
def test_compute_mean_prediction_and_confidence_mask(mc_samples, overlapping_classes):
    spatial_dim = [5, 5]
    num_classes = 3
    num_batch = 2
    example_logits = torch.randn([mc_samples, num_batch, num_classes, *spatial_dim])
    mean_logits = compute_mean_prediction(example_logits, overlapping_classes, mc_dim=0)
    confid = compute_confidence_map(
        example_logits,
        MaximumSoftmaxScore(),
        overlapping_classes=True,
        mc_dim=0,
    )
    # check the shapes
    assert mean_logits.shape == (num_batch, num_classes, *spatial_dim)
    assert confid.shape == (num_batch, *spatial_dim)
