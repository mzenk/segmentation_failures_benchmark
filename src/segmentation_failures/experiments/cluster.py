# import json
import re
import shlex
import subprocess
from pathlib import Path

from pssh.clients import SSHClient
from pssh.exceptions import Timeout
from rich import print
from rich.syntax import Syntax

from segmentation_failures.experiments.experiment import Experiment

BASH_BSUB_COMMAND = r"""
bsub -gpu num=1:j_exclusive=yes:gmem={gmem}\
    -L /bin/bash \
    -R "select[hname!='e230-dgx2-1']" \
    -q gpu \
    -u 'm.zenk@dkfz-heidelberg.de' \
    -N \
    -J "{name}" \
    -o $HOME/job_outputs/%J.out \
    -g /m167k/default_limited \
    bash -li -c 'set -o pipefail; echo $LSB_JOBID && source $HOME/.bashrc_segfail_new && {command}'
"""
# I use a .bashrc file here, which is easier on the cluster. dotenv won't override this

BASH_BSUB_COMMAND_CPU = r"""
bsub -q long \
    -n 8 \
    -R "rusage[mem=100G]" \
    -L /bin/bash \
    -u 'm.zenk@dkfz-heidelberg.de' \
    -N \
    -J "{name}" \
    -o $HOME/job_outputs/%J.out \
    bash -li -c 'set -o pipefail; echo $LSB_JOBID && source $HOME/.bashrc_segfail_new && {command}'
"""

BASH_BASE_COMMAND = r"""
HYDRA_FULL_ERROR=1 python $HOME/rsynced_code/segfail_project_new/src/segmentation_failures/scripts/{task}.py {overwrites}
"""
# task can be: train_seg, train_image_csf, test_fd

# Run before executing anything on the cluster
# later, it may be better to switch to a git repository and also log the commit hash
RSYNC_CODE_COMMAND = r"""
rsync -rtvu --delete --stats -f'- __pycache__/' -f'+ src/***' -f'+ pyproject.toml'  -f'- *' {source_dir}/ m167k@odcf-worker02.dkfz.de:/home/m167k/rsynced_code/segfail_project_new  # noqa B950
"""


def submit(
    _experiments: list[Experiment],
    task: str,
    dry_run: bool,
    user_overwrites: dict,
    cpu=False,
    gmem="10.7G",
):
    if len(_experiments) == 0:
        print("Nothing to run")
        return

    rsync_cmd = RSYNC_CODE_COMMAND.format(source_dir=Path(__file__).parents[3].absolute())
    print(
        Syntax(
            rsync_cmd.strip(),
            "bash",
            word_wrap=True,
            background_color="default",
        )
    )
    if dry_run:
        rsync_cmd = rsync_cmd.replace("rsync", "rsync -n", 1)
    subprocess.run(shlex.split(rsync_cmd), check=True)

    client = SSHClient("odcf-worker02.dkfz.de")
    for experiment in _experiments:
        # Compile overwrites. Precedence: cmdline > experiment > global
        final_overwrites = experiment.overwrites()
        final_overwrites["hydra"] = "cluster"  # affects expt folder naming
        final_overwrites.update(user_overwrites)
        overwrites = " ".join([f"'{k}={v}'" for k, v in final_overwrites.items()])
        cmd = BASH_BASE_COMMAND.format(
            task=task,
            overwrites=overwrites,
        ).strip()

        print(
            Syntax(
                re.sub(r"([^,]) ", "\\1 \\\n\t", cmd),
                "bash",
                word_wrap=True,
                background_color="default",
            )
        )

        if cpu:
            cmd = BASH_BSUB_COMMAND_CPU.format(
                name=f"{experiment.task} {experiment.dataset}_{experiment.name}",
                command=cmd,
            ).strip()
        else:
            cmd = BASH_BSUB_COMMAND.format(
                name=f"{experiment.task} {experiment.dataset}_{experiment.name}",
                command=cmd,
                gmem=gmem,
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
        try:
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
        except subprocess.CalledProcessError:
            continue
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
