import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import dotenv
from pssh.clients import SSHClient
from pssh.exceptions import Timeout
from rich import print
from rich.syntax import Syntax

from segmentation_failures.experiments.cluster import (
    BASH_BSUB_COMMAND_CPU,
    RSYNC_CODE_COMMAND,
)

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(Path(__file__).absolute().parents[1] / ".env", override=False, verbose=True)


BASH_LOCAL_COMMAND = r"""
bash -c 'set -o pipefail; {command} |& tee -a "$HOME/outputs_segfail/{log_file_name}.log"'
"""


def run_locally(cmd: str, dry_run: bool):
    log_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-prepare_auxdata"
    cmd = cmd.strip()
    print(
        Syntax(
            cmd,
            "bash",
            word_wrap=True,
            background_color="default",
        )
    )

    cmd = BASH_LOCAL_COMMAND.format(
        command=cmd,
        log_file_name=log_file_name,
    ).strip()
    if not dry_run:
        subprocess.run(shlex.split(cmd), check=True)


def submit(cmd: str, dry_run: bool, job_name: str = None, queue="medium", mem="64G"):
    if job_name is None:
        job_name = cmd
    rsync_cmd = RSYNC_CODE_COMMAND.format(source_dir=Path(__file__).parents[3].absolute())
    print(
        Syntax(
            rsync_cmd.strip(),
            "bash",
            word_wrap=True,
            background_color="default",
        )
    )
    if not dry_run:
        # rsync to cluster before execution
        subprocess.run(shlex.split(rsync_cmd), check=True)

    # run actual workload
    client = SSHClient("bsub01.lsf.dkfz.de")
    cmd = cmd.strip()
    print(
        Syntax(
            cmd,
            "bash",
            word_wrap=True,
            background_color="default",
        )
    )

    cmd = BASH_BSUB_COMMAND_CPU.format(
        name=job_name,
        command=cmd,
    ).strip()

    print(
        Syntax(
            cmd,
            "bash",
            word_wrap=True,
            background_color="default",
        )
    )

    if dry_run:
        return
    with client.open_shell(read_timeout=1) as shell:
        shell.run(cmd)

        try:
            for line in shell.stdout:
                print(line)
        except Timeout:
            pass

        try:
            for line in shell.stderr:
                print(line)
        except Timeout:
            pass


def main():
    parser = argparse.ArgumentParser(description="Prepare auxiliary data")
    parser.add_argument(
        "--expt_group",
        type=str,
        required=True,
        help="Name of the experiment group. Will look in $EXPERIMENT_ROOT_DIR for this name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset ID to prepare auxiliary data for.",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default=None,
        help="Path to src code directory (on the cluster if --cluster is specified).",
    )
    parser.add_argument(
        "--start_fold", type=int, default=0, help="Start fold for cross-validation."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        help="Backbone to use for segmentation model.",
        default="dynamic_unet_dropout",
    )
    parser.add_argument("--cluster", action="store_true", help="Run on cluster")
    parser.add_argument("--dry_run", action="store_true", help="Dry run")
    args = parser.parse_args()

    dataset = args.dataset
    code_dir = args.code_dir
    expt_group = args.expt_group
    dry_run = args.dry_run
    # just copied the dataset folder names
    ds_name_mapping = {
        "simple_fets22_corrupted": 500,
        "brats19_lhgg": 503,
        "mnms": 511,
        "mvseg23": 514,
        "kits23": 515,
        "covid_gonzalez": 520,
        "retina": 531,
        "prostate_gonzalez": 521,
        "retouch_cirrus": 540,
        "octa500": 560,
    }
    if dataset not in ds_name_mapping:
        raise ValueError(f"Unknown dataset: {dataset}")
    dataset_id = ds_name_mapping[dataset]
    if args.code_dir is None:
        if args.cluster:
            code_dir = "$HOME/rsynced_code/segfail_project_new/src/segmentation_failures"
        else:
            code_dir = Path(__file__).resolve().parents[1]

    template = r"""
    python {code_dir}/scripts/prepare_data_quality_regression.py \
        --dataset {dataset} \
        --experiment_root $EXPERIMENT_ROOT_DIR/{expt_group}/Dataset{dataset}/runs/{expt_name}/validate_pixel_csf \
        --folds {fold_str} \
        --seed 0 \
        --expt_group {expt_group}"""
    for pixel_csf in ["baseline", "deep_ensemble"]:
        # NOTE this requires nnunet raw data, which I usually don't store on the cluster.
        # Have to copy it first.
        expt_name = f"dynunet-{args.backbone}-{pixel_csf}"
        if dataset_id == 500:
            expt_name = f"baseline-monai_unet_dropout-{pixel_csf}"

        # quality regression preparation
        cmd = template.format(
            code_dir=code_dir,
            dataset=dataset_id,
            expt_group=expt_group,
            expt_name=expt_name,
            fold_str=" ".join([str(args.start_fold + i) for i in range(5)]),
        )
        job_name = f"Prepare CV-predictions for QR: {dataset_id} ({pixel_csf})"
        if pixel_csf != "deep_ensemble":
            # won't do quality regression training on ensemble predictions
            if args.cluster:
                submit(cmd, dry_run=dry_run, job_name=job_name)
            else:
                run_locally(cmd, dry_run=dry_run)
        print("--------------------")

        # heuristic/radiomics preparation
        cmd = template.format(
            code_dir=code_dir,
            dataset=dataset_id,
            expt_group=expt_group,
            expt_name=expt_name,
            fold_str=" ".join([str(args.start_fold + i) for i in range(5)]),
        )
        if pixel_csf == "deep_ensemble":
            cmd += " --confid predictive_entropy mutual_information"
            # cmd += " --confid predictive_entropy"
        elif pixel_csf == "baseline":
            cmd += " --confid predictive_entropy"
        cmd += " --imagecsf_name heuristic_radiomics --no_preprocessing"
        job_name = f"Prepare CV-predictions for Heur./Rad.: {dataset_id} ({pixel_csf})"
        if args.cluster:
            submit(cmd, dry_run=dry_run, job_name=job_name)
        else:
            run_locally(cmd, dry_run=dry_run)


if __name__ == "__main__":
    main()
