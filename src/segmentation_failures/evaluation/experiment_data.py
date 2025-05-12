from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, ListConfig

from segmentation_failures.utils.io import load_expt_config, load_json, save_json


# Interface to failure detection analysis
@dataclass
class ExperimentData:
    case_ids: List[str]  # Length n_samples. For reidentifying individual cases
    domain_names: List[str]  # Length n_samples; domain of each sample (for multi-domain datasets)
    confid_scores: npt.NDArray[Any]
    # Shape (n_samples, n_scores). Note that n_scores can vary between experiments; usually it's 1.
    confid_scores_names: List[str]  # Length n_scores
    segmentation_metrics: npt.NDArray[Any]  # Shape (n_samples, n_metrics)
    segmentation_metrics_multi: npt.NDArray[Any]
    # Multi-class metrics; Shape (n_samples, n_classes, n_metrics_multi)
    segmentation_metrics_names: List[str]  # Length n_metrics
    segmentation_metrics_names_multi: List[str]  # Length n_metrics_multi
    config: (
        DictConfig | ListConfig
    )  # Do I really need this? If I remove it, I should keep the task (dataset) ID at least.
    # task id is contained in config.

    def validate(self):
        # could add further checks
        assert self.confid_scores.shape[1] == len(
            self.confid_scores_names
        ), f"Should be identical: {len(self.confid_scores_names)} vs. {self.confid_scores.shape[1]}"
        assert self.segmentation_metrics.shape[1] == len(
            self.segmentation_metrics_names
        ), f"Should be idenctical: {len(self.segmentation_metrics_names)} vs. {self.segmentation_metrics.shape[1]}"

    def to_dataframe(self) -> pd.DataFrame:
        result_df = pd.DataFrame(index=list(range(len(self.confid_scores))), columns=[])
        for idx, score in enumerate(self.confid_scores_names):
            result_df[f"confid_{score}"] = self.confid_scores[:, idx]
        result_df["case_id"] = self.case_ids
        result_df["domain"] = self.domain_names

        for idx in range(self.segmentation_metrics.shape[1]):
            metric_name = self.segmentation_metrics_names[idx]
            result_df[f"metric_{metric_name}"] = self.segmentation_metrics[:, idx]
        if len(self.segmentation_metrics_names_multi) > 0:
            for idx in range(self.segmentation_metrics_multi.shape[2]):
                for class_idx in range(self.segmentation_metrics_multi.shape[1]):
                    metric_name = self.segmentation_metrics_names_multi[idx]
                    result_df[f"metric_class{class_idx:02d}_{metric_name}"] = (
                        self.segmentation_metrics_multi[:, class_idx, idx]
                    )
        return result_df

    @staticmethod
    def __load_npz_if_exists(path: Path) -> npt.NDArray[np.float64] | None:
        if not path.is_file():
            return None
        with np.load(path) as npz:
            return npz.f.arr_0

    @staticmethod
    def __load_json_if_exists(path: Path) -> Any | None:
        if not path.is_file():
            return None
        return load_json(path)

    def save(self, save_dir: Path) -> None:
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        logger.info(f"Saving experiment data to: {save_dir.absolute()}")
        np.savez_compressed(save_dir / "confidence_scores.npz", self.confid_scores)
        np.savez_compressed(save_dir / "metrics.npz", self.segmentation_metrics)
        np.savez_compressed(save_dir / "metrics_multi.npz", self.segmentation_metrics_multi)
        save_json(self.confid_scores_names, save_dir / "confid_names.json")
        save_json(self.case_ids, save_dir / "case_ids.json")
        save_json(self.segmentation_metrics_names, save_dir / "metrics_names.json")
        save_json(self.segmentation_metrics_names_multi, save_dir / "metrics_names_multi.json")
        save_json(self.domain_names, save_dir / "domains.json")

    @staticmethod
    def load(save_dir: Path, config: DictConfig | ListConfig | None = None) -> ExperimentData:
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        logger.info(f"Loading experiment data from: {save_dir.absolute()}")
        confids = ExperimentData.__load_npz_if_exists(save_dir / "confidence_scores.npz")
        metrics = ExperimentData.__load_npz_if_exists(save_dir / "metrics.npz")
        metrics_multi = ExperimentData.__load_npz_if_exists(save_dir / "metrics_multi.npz")
        confid_names = ExperimentData.__load_json_if_exists(save_dir / "confid_names.json")
        case_ids = ExperimentData.__load_json_if_exists(save_dir / "case_ids.json")
        metrics_names = ExperimentData.__load_json_if_exists(save_dir / "metrics_names.json")
        metrics_names_multi = ExperimentData.__load_json_if_exists(
            save_dir / "metrics_names_multi.json"
        )
        domain_names = ExperimentData.__load_json_if_exists(save_dir / "domains.json")
        data = ExperimentData(
            config=config,
            confid_scores=confids,
            confid_scores_names=confid_names,
            case_ids=case_ids,
            segmentation_metrics=metrics,
            segmentation_metrics_multi=metrics_multi,
            segmentation_metrics_names=metrics_names,
            segmentation_metrics_names_multi=metrics_names_multi,
            domain_names=domain_names,
        )
        data.validate()
        return data

    @staticmethod
    def from_experiment_dir(
        expt_dir: Path,
    ) -> ExperimentData:
        if not isinstance(expt_dir, Path):
            expt_dir = Path(expt_dir)

        config = load_expt_config(expt_dir)
        # Since we're running this without hydra, I need to replace the output_dir path manually
        # (otherwise I get UnsupportedInterpolationType error)
        config.paths.output_dir = expt_dir.absolute()
        return ExperimentData.load(Path(config.paths.results_dir), config)
