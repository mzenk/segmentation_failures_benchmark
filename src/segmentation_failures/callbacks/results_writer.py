"""
I want to have two callbacks here:
1. Save predictions for the testing/prediction loop
2. Save confidences and segmentation metrics for the testing loop
    [may separate confidences later but since it's just a scalar for now I prefer this]
"""

from pathlib import Path

import numpy as np
import torch

# import torch.nn.functional as F
from loguru import logger
from pytorch_lightning.callbacks import Callback

from segmentation_failures.evaluation.failure_detection.fd_analysis import (
    ExperimentData,
)
from segmentation_failures.evaluation.segmentation.compute_seg_metrics import (
    compute_metrics_for_prediction_dir,
)


class ExperimentDataWriter(Callback):
    # at the end of test set evaluation, create and store the ExperimentData (= all segmentation metrics and confidence scores)
    def __init__(
        self,
        output_dir: str,
        num_classes: int,
        prediction_dir: str,
        metric_list: list[str] | None = None,
        region_based_eval=False,
        num_processes: int = 1,
        previous_stage_results_path: str | None = None,
    ):
        self.output_path = Path(output_dir)
        self.output_path.mkdir()
        self.prediction_dir = Path(prediction_dir)
        self.num_classes = num_classes
        self.region_based_eval = region_based_eval
        self.metrics = metric_list
        self.buffer_confidences = []
        self.buffer_confidences_multiclass = []
        self.buffer_confidence_names = []
        self.buffer_case_ids = []
        logger.debug(f"Using {num_processes} processes for metric computation.")
        self.num_processes = num_processes
        self.previous_stage_results_path = previous_stage_results_path

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        case_ids = batch["keys"]
        if "confidence" not in outputs:
            logger.warning(
                "The model did not return confidence scores. Inserting NaNs as image-level confidence scores."
            )
            confidence = {"none": torch.nan * torch.ones(len(batch["keys"]))}
        else:
            confidence = outputs["confidence"]  # dict [str, torch.Tensor with shape (num_batch,)]
        if isinstance(confidence, torch.Tensor):
            # for compatibility with old interface
            confidence = {"confid": confidence}
        for k, v in confidence.items():
            if v.dim() > 1:
                logger.warning(
                    f"Expected confidence of shape (num_batch,), but found a confidence in output with shape {v.shape}."
                    "Maybe pixel confidence maps were returned by the method? Inserting NaNs as image-level confidence scores."
                )
                confidence[k] = torch.nan * torch.ones(v.shape[0], device=v.device)

        sorted_confid_names = sorted(confidence)
        if len(self.buffer_confidence_names) == 0:
            self.buffer_confidence_names = sorted_confid_names
        elif self.buffer_confidence_names != sorted_confid_names:
            raise KeyError("Incompatible confidence score names obtained from different batches!")
        self.buffer_confidences.append(
            torch.stack([confidence[k] for k in sorted_confid_names], dim=1).cpu().numpy()
        )  # Shape (batch, n_confid)
        self.buffer_case_ids.extend(case_ids)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        test_data_dicts = trainer.datamodule.get_test_data_dicts()
        label_file_list = [Path(x["target"]) for x in test_data_dicts]
        assert all(x.is_file() for x in label_file_list)
        case_ids = [x["keys"] for x in test_data_dicts]
        domains = [x["domain_label"] for x in test_data_dicts]
        if len(self.buffer_case_ids) < len(case_ids):
            logger.warning("Not all cases were evaluated. Inserting NaNs for missing cases.")
        recompute_metrics = True
        if self.previous_stage_results_path is not None:
            logger.info("Loading previous stage results.")
            previous_stage_results = ExperimentData.load(self.previous_stage_results_path)
            recompute_metrics = False
            # check if all cases are present in the previous stage results
            previous_case_ids = previous_stage_results.case_ids
            if set(case_ids) != set(previous_case_ids):
                raise ValueError(
                    "The case IDs from the previous stage results do not match the current case IDs!"
                )
            # reorder the previous stage results to match the current case order
            previous_case_order = [previous_case_ids.index(x) for x in case_ids]
            metrics_arr = previous_stage_results.segmentation_metrics[previous_case_order]
            multi_metrics_arr = previous_stage_results.segmentation_metrics_multi[
                previous_case_order
            ]
            metric_names = previous_stage_results.segmentation_metrics_names
            multi_metric_names = previous_stage_results.segmentation_metrics_names_multi
            if not set(self.metrics).issubset(
                previous_stage_results.segmentation_metrics_names
                + previous_stage_results.segmentation_metrics_names_multi
            ):
                recompute_metrics = True
                self.prediction_dir = Path(self.previous_stage_results_path) / "predictions"
        if recompute_metrics:
            logger.info("Computing segmentation metrics for the test set.")
            metrics_dict, multi_metrics_dict = compute_metrics_for_prediction_dir(
                metric_list=self.metrics,
                prediction_dir=self.prediction_dir,
                label_file_list=label_file_list,
                dataset_id=trainer.datamodule.dataset_id,
                num_processes=self.num_processes,
            )
            # dicts with shape (n_samples,) and (n_samples, n_classes), respectively.
            assert all(arr.shape[0] == len(case_ids) for arr in metrics_dict.values())
            metric_names = list(metrics_dict.keys())
            multi_metric_names = list(multi_metrics_dict.keys())
            metrics_arr = np.stack([metrics_dict[k] for k in metrics_dict], axis=-1)
            if len(multi_metrics_dict) > 0:
                assert all(arr.shape[0] == len(case_ids) for arr in multi_metrics_dict.values())
                multi_metrics_arr = np.stack(
                    [multi_metrics_dict[k] for k in multi_metrics_dict], axis=-1
                )
            else:
                multi_metrics_arr = np.array([])

        # this assumes confidence shape (num_samples, [num_scores]) for each in the buffer
        confid_arr = np.ones((len(case_ids), len(self.buffer_confidence_names))) * np.nan
        confid_arr[[case_ids.index(x) for x in self.buffer_case_ids]] = np.concatenate(
            self.buffer_confidences, axis=0
        )
        # Save results
        results = ExperimentData(
            case_ids=case_ids,
            confid_scores=confid_arr,
            confid_scores_names=self.buffer_confidence_names,
            segmentation_metrics=metrics_arr,
            segmentation_metrics_names=metric_names,
            segmentation_metrics_multi=multi_metrics_arr,
            segmentation_metrics_names_multi=multi_metric_names,
            domain_names=domains,
            config=None,  # we don't need it here (isn't saved)
        )
        results.save(self.output_path)
        self.buffer_case_ids = []
        self.buffer_confidences = []
