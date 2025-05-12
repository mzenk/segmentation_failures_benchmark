from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from segmentation_failures.evaluation.experiment_data import ExperimentData
from segmentation_failures.utils.io import load_expt_config


def load_raw_results(expt_dir):
    expt_data = ExperimentData.from_experiment_dir(Path(expt_dir))
    expt_config = expt_data.config
    expt_df = expt_data.to_dataframe()
    expt_df["expt_name"] = expt_data.config.expt_name
    if "hparams" in expt_config.datamodule:
        expt_df["fold"] = expt_config.datamodule.hparams.fold
    else:
        expt_df["fold"] = expt_config.datamodule.fold
    expt_df["seed"] = expt_config.seed
    return expt_df


def load_fd_results_hydra(
    expt_root_dir: Path,
    csv_name="fd_metrics.csv",
    legacy_structure=False,
) -> Tuple[pd.DataFrame, DictConfig]:
    # assumes expt_root_dir structure: $expt_root_dir/experiment_name/test_fd/<run_dir>
    # TODO maybe select the csv file with highest count (e.g. fd_metrics_1.csv, fd_metrics_2.csv, ...)
    # because this is how it is saved when rerunning the evaluation
    all_results = []
    all_configs = {}
    expt_id = 0
    for expt_dir in expt_root_dir.iterdir():
        test_runs_root = expt_dir / "test_fd"
        if legacy_structure:
            test_runs_root = expt_dir
        if not test_runs_root.exists():
            continue
        for run_version_dir in test_runs_root.iterdir():
            logger.debug(run_version_dir)
            try:
                results, config = load_single_fd_result_hydra(run_version_dir, csv_name)
            except FileNotFoundError:
                logger.warning(
                    f"Could not find results file {csv_name} in {run_version_dir}. Ignoring this!"
                )
            else:
                results["expt_id"] = expt_id
                all_results.append(results)
                all_configs[expt_id] = config
            expt_id += 1
            # it could happen that we don't find the csv for this run, but we still want to count it,
            # because there might be other csvs
    if len(all_results) > 0:
        return pd.concat(all_results, ignore_index=True), all_configs
    else:
        logger.warning(f"Couldn't find experiment runs here: {expt_root_dir}")
        return pd.DataFrame(), DictConfig({})


def load_single_fd_result_hydra(expt_dir: Path, csv_name: str):
    if not csv_name.lower().endswith(".csv"):
        csv_name += ".csv"
    # load run configuration
    config = load_expt_config(expt_dir, resolve=False)
    # I set the output dir manually here because we already know it (don't need hydra).
    # Otherwise, there can be issues when running on cluster/analysing on workstation
    config.paths.output_dir = str(expt_dir)
    if "analysis_dir" not in config.paths:
        raise RuntimeError(
            "No analysis_dir entry in config.paths found. Seems like the directory follows another storing convention."
        )
    results_csv = Path(config.paths.analysis_dir) / csv_name
    if not results_csv.exists():
        raise FileNotFoundError
    results = pd.read_csv(results_csv, index_col=0)
    results["root_dir"] = str(expt_dir.absolute())
    return results, config
