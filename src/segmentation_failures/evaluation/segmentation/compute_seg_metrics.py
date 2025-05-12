import multiprocessing
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from loguru import logger
from nnunetv2.imageio.reader_writer_registry import (
    determine_reader_writer_from_dataset_json,
)
from nnunetv2.utilities.label_handling.label_handling import LabelManager

from segmentation_failures.evaluation.segmentation.distance_thresholds import (
    get_distance_thresholds,
)
from segmentation_failures.evaluation.segmentation.segmentation_metrics import (
    get_metrics,
    get_metrics_info,
)
from segmentation_failures.utils.data import load_dataset_json
from segmentation_failures.utils.label_handling import convert_to_onehot


def compute_metrics_for_file(
    metric_obj_dict: dict,
    label_file: Path,
    pred_file: Path,
    all_labels: list,  # Important! not dict
    seg_reader_fn: Callable,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute metrics for a single label and prediction file pair.

    The reason why this is not a method of the ExperimentDataWriter class is that I had issues with
    multiprocessing (pickling error). Idk exactly why but suspect that there was some reference in
    the callback that was not picklable.
    """
    num_classes = len(all_labels)
    if isinstance(all_labels, dict):
        # I could also convert here, but maybe it's better to just raise an error
        raise TypeError("all_labels must be a list, not a dict!")
    if not label_file.exists():
        raise FileNotFoundError(f"Label file {label_file} does not exist!")
    else:
        target, target_properties = seg_reader_fn(label_file)
        label_spacing = target_properties["spacing"]
    if not pred_file.exists():
        pred, pred_spacing = None, None
    else:
        pred, pred_properties = seg_reader_fn(pred_file)
        pred_spacing = pred_properties["spacing"]
    result = {}
    result_multi = {}
    if target is None:
        raise TypeError("Can't evaluate with missing label!")
    if pred is None:
        # TODO this is a bit unnecessary, but works. Maybe improve later
        # insert NaN metrics; should only happen during debugging runs
        for metric_name, metric_obj in metric_obj_dict.items():
            result[metric_name] = metric_obj(
                torch.ones((1, num_classes) + target.shape[-3:]) * torch.nan,
                torch.ones((1, num_classes) + target.shape[-3:]) * torch.nan,
            ).numpy()[
                0
            ]  # remove batch dimension
    else:
        if not np.all(pred.astype(int) == pred):
            raise TypeError("Predictions must be integer!")
        if pred.shape != target.shape:
            # NOTE this is a breaking change. Fails if the files in the label dir and prediction dir have different spacing
            raise ValueError(
                f"Shape mismatch between prediction and label for {label_file.name}: {pred.shape} vs {target.shape}"
            )
        if pred_spacing is not None and not np.all(np.isclose(pred_spacing, label_spacing)):
            raise ValueError(
                f"Spacing mismatch between prediction and label for {label_file.name}: {pred_spacing} vs {label_spacing}"
            )
        # label shape, pred_shape: (1, H, W[, D])
        # This handles both cases where all_labels is region-based (nnunet style) or exclusive
        target = np.swapaxes(convert_to_onehot(target, all_labels), 0, 1)
        pred = np.swapaxes(convert_to_onehot(pred, all_labels), 0, 1)
        for metric_name, metric_obj in metric_obj_dict.items():
            kwargs = {}
            if metric_name in ["hausdorff95", "surface_dice"]:
                kwargs = {"spacing": label_spacing}
                if isinstance(kwargs["spacing"], np.ndarray):
                    # monai cannot use float32
                    kwargs["spacing"] = kwargs["spacing"].astype(float)
            result[metric_name] = metric_obj(
                torch.from_numpy(pred), torch.from_numpy(target), **kwargs
            ).numpy()[
                0
            ]  # remove batch dimension
    # check if there are a multi-class metric and move them to the multi-metric dict
    new_result = {}
    metric_infos = get_metrics_info(list(metric_obj_dict.keys()))
    for metric_name, metric_val in result.items():
        if metric_infos[metric_name].classwise:
            # multiclass metric case
            result_multi[metric_name] = metric_val
            # this can be nan if there are nans in the metric:
            new_result[f"mean_{metric_name}"] = metric_val.mean().reshape((1,))
        else:
            new_result[metric_name] = np.reshape(metric_val, (1,))  # avoid 0D arrays
    result = new_result
    return result, result_multi


def compute_metrics_for_prediction_dir(
    metric_list,
    prediction_dir: str,
    label_file_list: list[Path],
    dataset_id: int,
    num_processes: int = 1,
):
    # I pass a list of label files so that I automatically get an ordering of test cases
    prediction_dir = Path(prediction_dir)
    dataset_json = load_dataset_json(dataset_id)
    lm = LabelManager(
        dataset_json["labels"],
        regions_class_order=dataset_json.get("regions_class_order"),
    )
    # I include the BG label because my metrics function excludes it
    # TODO or just always remove the background?
    if lm.has_regions:
        # There is a bug in nnunetv2 that excludes the background here
        # TODO remove this special treatment once the bug is fixed
        all_labels = lm.all_regions
        if 0 not in all_labels and (0,) not in all_labels:
            all_labels = [0] + all_labels
        num_fg_labels = len(lm.foreground_regions)
    else:
        all_labels = lm.all_labels
        num_fg_labels = len(lm.foreground_labels)

    example_file = label_file_list[0]
    nnunet_reader = determine_reader_writer_from_dataset_json(dataset_json, str(example_file))()
    reader_fn = nnunet_reader.read_seg

    pred_file_list = []
    file_ending = dataset_json.get("file_ending", ".nii.gz")
    for lfile in label_file_list:
        case_id = lfile.name.removesuffix(file_ending)
        pred_file_list.append(prediction_dir / f"{case_id}{file_ending}")

    # initialize metric objects
    metric_obj_dict = {}
    for metric_name in metric_list:
        # metrics should disregard the background class
        kwargs = {"include_background": False}
        if metric_name in ["surface_dice"]:
            kwargs["class_thresholds"] = get_distance_thresholds(dataset_id)
        metric_obj_dict.update(get_metrics(metric_name, **kwargs))
    # iterate through the label and prediction files and compute metrics for each pair
    all_metric_args = []
    for label_file, pred_file in zip(label_file_list, pred_file_list):
        all_metric_args.append(
            (
                metric_obj_dict,
                label_file,
                pred_file,
                all_labels,
                reader_fn,
            )
        )
    start_time = time.time()
    if num_processes <= 1:
        results = []
        for args in all_metric_args:
            results.append(compute_metrics_for_file(*args))
    else:
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            results = pool.starmap(compute_metrics_for_file, all_metric_args)
    logger.debug(f"Metric computation took {time.time() - start_time} seconds.")

    # collect and organize results
    metric_results = defaultdict(list)
    multimetric_results = defaultdict(list)
    for results_entry in results:
        # I currently add mean_X (X=Dice etc) manually in the _compute_metrics_for_file, so they are not in
        # metric_list. TODO in the future, it may be better to have a separate metric entry for mean dice etc
        curr_metrics, curr_multimetrics = results_entry
        for metric_name, metric_val in curr_metrics.items():
            assert metric_val.shape == (1,)
            metric_results[metric_name].append(metric_val)
        for metric_name, metric_val in curr_multimetrics.items():
            assert metric_val.shape == (num_fg_labels,)
            multimetric_results[metric_name].append(metric_val)

    metric_results = {k: np.concatenate(v) for k, v in metric_results.items()}
    multimetric_results = {k: np.stack(v, axis=0) for k, v in multimetric_results.items()}
    return metric_results, multimetric_results


if __name__ == "__main__":
    # testing
    metric_list = ["dice", "surface_dice"]
    # label_dir = "/home/m167k/Datasets/segmentation_failures/nnunet_convention_new/Dataset500_simple_fets_corruptions/labelsTr"
    # dataset_json = "/home/m167k/Datasets/segmentation_failures/nnunet_convention_new/Dataset500_simple_fets_corruptions/dataset.json"  # noqa B950
    label_dir = Path(
        "/home/m167k/Datasets/segmentation_failures/nnunet_convention_new/Dataset503_BraTS19/labelsTs"
    )
    dataset_json = "/home/m167k/Datasets/segmentation_failures/nnunet_convention_new/Dataset503_BraTS19/dataset.json"
    # maybe also test with a prediction dir that lacks some files later
    results = compute_metrics_for_prediction_dir(
        metric_list=metric_list,
        prediction_dir=str(label_dir),
        dataset_id=503,
        label_file_list=list(label_dir.glob("*.nii.gz")),
        num_processes=1,
    )
