"""How to test?
- Test that rejector model works as expected: Generate dummy data (arbitrary regression task), run fit and predict with pipeline
- Test extract_features
"""

import time

import torch

from segmentation_failures.models.confidence_aggregation import (
    ForegroundAggregator,
    ForegroundSizeAggregator,
    HeuristicAggregationModule,
)
from segmentation_failures.models.confidence_aggregation.base import (
    PairwiseDiceAggregator,
)


def test_extract_features():
    dummy_module = HeuristicAggregationModule(
        regression_model="regression_forest",
        dataset_id=500,
        confid_name="dummy_confid",
        target_metrics=["generalized_dice"],
        heuristic_list=[ForegroundAggregator(), ForegroundSizeAggregator()],
    )
    # region based is a bit hard to simulate here
    dummy_prediction = torch.tensor(
        [
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
    )
    dummy_prediction = dummy_prediction.reshape(1, 1, *dummy_prediction.shape)
    dummy_confid = torch.rand_like(dummy_prediction, dtype=float)
    features = dummy_module.extract_features(dummy_prediction, dummy_confid)
    assert features.shape == (len(dummy_prediction), len(dummy_module.aggregator_list))


def test_pairwise_dice_agg(num_batch=4, img_size=(5, 5), region_based=True):
    NUM_CLASSES = 2
    NUM_SAMPLES = 4
    consensus_pred = torch.zeros(NUM_SAMPLES, num_batch, NUM_CLASSES, *img_size)
    start_x = 1
    start_y = 1
    size = 2
    consensus_pred[:, :, 1, start_x : start_x + size, start_y : start_y + size] = 1
    if region_based:
        consensus_pred[:, :, 0, 0:size, -size:] = 1
    consensus_pred = consensus_pred.to(dtype=torch.bool)
    score = PairwiseDiceAggregator(include_zero_label=region_based)
    start = time.time()
    result = score.aggregate(consensus_pred)
    end = time.time()
    print(f"Time taken for pairwise dice: {end - start} seconds")
    assert torch.allclose(result, torch.ones_like(result))

    disjoint_pred = torch.zeros(NUM_SAMPLES, num_batch, NUM_CLASSES, *img_size)
    for i in range(len(disjoint_pred)):
        disjoint_pred[i, :, 1, i] = 1
        disjoint_pred[i, :, 0] = 1 - disjoint_pred[i, :, 1]
        if region_based:
            disjoint_pred[i, :, 0] = disjoint_pred[i, :, 1]

    disjoint_pred = disjoint_pred.to(dtype=torch.bool)
    result = score.aggregate(disjoint_pred)
    assert torch.allclose(result, torch.zeros_like(result))
