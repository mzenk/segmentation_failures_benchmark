Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its [license](LICENSE).

# Failure Detection Methods in Medical Image Segmentation

This repository is the official implementation of the paper [Comparative Benchmarking of Failure Detection Methods in Medical Image Segmentation: Unveiling the Role of Confidence Aggregation](https://doi.org/10.1016/j.media.2024.103392).

<div align="center">
<img src=assets/fd_figure1.svg width="600px">
</div>

## Installation

This package (`segmentation_failures`) requires Python version 3.10 or later.

1. Clone this repository and `cd` into the root directory (where this readme is).
1. Optional but strongly recommended: Create a new python environment (venv, conda environment, ...) and activate it.
1. Install an appropriate version of [PyTorch](https://pytorch.org/get-started/locally/). Check that CUDA is available and that the CUDA toolkit version is compatible with your hardware. The currently necessary version of pytorch is 2.0.1 ([installation instructions](https://pytorch.org/get-started/previous-versions/#v201)). Testing and Development was done with the pytorch version using CUDA 11.7.
1. Install `segmentation_failures` locally. This will pull in all dependencies and keep the PyTorch version you installed previously (or update it if it differs from the requirements).

    ```bash
    pip install -e '.[dev,launcher]'
    ```

<!-- Offer an alternative installation via git? see [here](https://pip.pypa.io/en/stable/cli/pip_install/). -->

## How to reproduce our results

### Set environment variables

The following environment variables are needed to use the full functionality of this repository. Please make sure the corresponding directories exist:

```bash
export TESTDATA_ROOT_DIR=/path/to/test/datasets
export SEGFAIL_AUXDATA=/path/to/auxiliary/data
export EXPERIMENT_ROOT_DIR=/path/where/experiment/logs/are/saved
```

A short explanation:

- `TESTDATA_ROOT_DIR` is where test datasets are stored (more on this in [Prepare datasets](#prepare-the-datasets))
- `SEGFAIL_AUXDATA` is used to store auxiliary training data for the quality regression method
- `EXPERIMENT_ROOT_DIR` is used to store all experiment results and checkpoints

### Prepare the datasets

To download the raw data, please follow the instructions on the respective websites:

- **Brain tumor (2D):** : The raw data can be obtained from the FeTS challenge 2022 [website](https://www.synapse.org/#!Synapse:syn28546456/wiki/617093).
- **Brain tumor:**: This dataset uses data from the BraTS challenge 2019, which has information on the tumor grade. Access can be gained after registering [here](https://www.med.upenn.edu/cbica/brats2019/registration.html).
- **Heart:** Fill out the form that can be found in the Download subpage to get access. [Link](https://www.ub.edu/mnms/)
- **Kidney tumor:** The data can be downloaded from their [git repository](https://github.com/neheller/kits23?tab=readme-ov-file#usage).
- **Covid:** This collection consists of data from a challenge and two other sources. The challenge data can be downloaded through [grand-challenge.org](https://covid-segmentation.grand-challenge.org/COVID-19-20/) after registration. The radiopaedia subset can be downloaded from [zenodo](https://zenodo.org/records/3757476). The website for the MosMed dataset is unfortunately not available anymore, but the data can be shared upon request.
- **Prostate:** This collection is a combination of Medical Segmentation Decathlon (MSD) data and additional datasets for testing. The MSD data can be downloaded from [their webpage](http://medicaldecathlon.com/) (Task05). The testing datasets can be downloaded through [this webpage](https://liuquande.github.io/SAML/). Note that we exclude the RUNMC cases, as they might overlap with the MSD data.

As we make use of nnU-Net functions for data preparation and preprocessing, make sure you have set up the [paths it needs](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md), in particular `nnUNet_raw` and `nnUNet_preprocessed`.

After obtaining the raw data (and unzipping them, if necessary), they have to be converted to a common format, which follows nnU-Net. All scripts used for our experiments are available in [this folder](./src/segmentation_failures/data/dataset_conversion/). For the datasets used in the paper, the scripts `nnunet_{simple_fets_corruptions,brats19,mnm,kits23,covid,prostate}.py` are relevant. Please execute the script for each dataset you want to use (they have slightly different input paths), for example

```bash
python nnunet_mnm.py /path/to/downloaded/data_mnms --use_default_splits
```

The argument `--use_default_splits` makes sure that the same train-test splits as in our paper are used.

To preprocess the datasets (except brain tumor 2D), simply run

```bash
nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 3d_fullres --verify_dataset_integrity
```

This might take a while, especially for the kidney tumor dataset. For the brain tumor (2D) dataset, conversion and preprocessing are combined in the conversion script, so you do not need to execute the nnunet command for it.

For test set evaluation, the images, labels and domain mappings have to be accessible through the `TESTDATA_ROOT_DIR` variable. To achieve this, either point it to the nnU-Net raw data directory (`export TESTDATA_ROOT_DIR=$nnUNet_raw`) or copy the contents for each dataset (except `imagesTr` and `labelsTr`) to `$TESTDATA_ROOT_DIR/$DATASET_NAME`, in case testing data should be stored on different storage devices.

Final check: To make sure you use the same splits as we did, please run the script [`check_dataset_splits.py`](src/segmentation_failures/scripts/check_dataset_splits.py). This will copy the training-validation splits from [here](./dataset_splits/) to your dataset directory and also check that the test set consists of the same cases as in our experiments.

### Run experiments

All experiments from our paper were started using the script [`experiments_paper.sh`](./src/segmentation_failures/experiments/experiments_paper.sh), which calls the experiment launcher for multiple methods and/or datasets. Due to the diversity of failure detection methods, running all experiments is a multi-step procedure:

1. `cd src/segmentation_failures/experiments`
1. Train all segmentation networks (`# SEGMENTATION` in `experiments_paper.sh`)
1. If quality regression methods should be evaluated, their training data have to be prepared:
    1. Obtain CV predictions on all datasets (`# CROSS-VALIDATION PIXEL CSF`).
    1. Prepare training data, i.e. model segmentations (`# PREPARE AUXDATA`)
1. Train two-stage failure detection methods (`# STAGE 2 TRAINING`) like quality regression or Mahalanobis
1. Run inference for segmentation and pixel CSF methods (`# INFERENCE PIXEL CSF`)
1. Test all methods on the failure detection task (`# FAILURE DETECTION TESTING`)

The corresponding sections in `experiments_paper.sh` have to be uncommented and run in above order.

### Visualize results

After all experiments are run, the following table can be produced to recover our results:

<div align="center">
<img src=assets/paper_table.png width="800px">
</div>

To get this table (and all other figures from our paper), you can run the [analysis script](./analysis/paper_analysis.py).

## How to use this repository for your own research

Below are some examples how you could use this repository for your own work.

### Evaluate your own experiments with AURC

If you just want to use our evaluation pipeline independently of the datasets and methods from this benchmark, you can check out the metrics implementation in [`metrics.py`](src/segmentation_failures/evaluation/failure_detection/metrics.py). Here is a simple example for how this can be used

```python
import numpy as np
from segmentation_failures.evaluation.failure_detection.metrics import StatsCache, get_metric_function
# compute risk scores and confidence scores as you like
risks = np.random.rand(100)
confids = np.random.rand(100)

stats = StatsCache(confids=confids, risks=risks)
aurc_values = get_metric_function("aurc")(stats)
```

A more elaborate example is given by the evaluation pipeline used for this benchmark, which you can find [here](src/segmentation_failures/evaluation/failure_detection/fd_analysis.py)

### Use the datasets for your own experiments

If you just want to use our dataset setup independently of the experiments implementation from this benchmark, just prepare the datasets as described [above](#prepare-the-datasets).

### Overwrite the configuration via CLI and the experiment launcher

(Note: This assumes you work in the code framework of this benchmark)

We use [hydra](https://hydra.cc/docs/intro/) for configuring experiments. The basic configuration is contained in the [configuration folder](src/segmentation_failures/conf) and it is extended by special overwrites in the [experiment launcher](src/segmentation_failures/experiments/__init__.py) (see the `overwrites` function). Any configuration value can, however, be simply modified via the command line using the `--overwrites` argument of the experiment launcher. Here are some examples:

- Adapt the learning rate: `python launcher.py --task train_seg --dataset mnms --fold 0 --seed 0 --backbone dynamic_unet_dropout --overwrites segmentation.hparams.lr=1e-3`
- Use a different checkpoint for evaluation: `python launcher.py --task test_fd --dataset mnms --fold 0 --seed 0 --backbone dynamic_unet_dropout --csf_pixel baseline --csf_image quality_regression --overwrites csfimage.checkpoint=/path/to/ckpt/last.ckpt`

The launcher ultimately builds the commands for the scripts in [this folder](src/segmentation_failures/scripts), so alternatively you can also use these directly.

### Add more datasets

(Note: This assumes you work in the code framework of this benchmark)

1. Add datasets in nnunet format (otherwise, customized data loaders are needed) and save the preparation script to [src/segmentation_failures/data/dataset_conversion](src/segmentation_failures/data/dataset_conversion).
1. Preprocessing using nnU-Net: `nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 3d_fullres --verify_dataset_integrity`
1. Extend the configuration:
    - In the `dataset` group, add meta-information about the dataset.
    - In the `datamodule` group, add an entry with nnunet as default and inserting patch and batch size
    - if you want to use quality regression and vae, add entry to `HARDCODED_IMG_SIZES` dict.
1. Extend the [experiment launcher](src/segmentation_failures/experiments/__init__.py): Add the dataset to the standard experiments to make it launchable (functions called by `get_experiments`).

Now you should be ready to train and test models on your new dataset using the launcher.

### Add more failure detection methods

(Note: This assumes you work in the code framework of this benchmark)

1. Add your implementation in an appropriate subfolder of `src/segmentation_failures/models`. Note: Make sure your model is compatible with the datamodule you intend to use. If none of the [existing modules](src/segmentation_failures/data/datamodules) are useful, you'll have to implement your own datamodule.
1. Extend configuration: Add a config file to `csf_{aggregation,image,pixel}` depending on the kind of method and make sure your new method class is used in `_target_`. Also, add any dataset-independent configuration options in that file.
1. Extend the [experiment launcher](src/segmentation_failures/experiments/__init__.py): Add the method to the standard experiments to make it launchable (functions called by `get_experiments`). If there are any dataset-specific overwrites, extend the `overwrites` function (or pass them as manual overwrites like [here](#overwrite-the-configuration-via-cli-and-the-experiment-launcher)).

<!-- ## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z.

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. -->

<!-- ## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.  -->

## Acknowledgements

The AURC implementation and many other parts were adapted from [fd-shifts](https://github.com/IML-DKFZ/fd-shifts). Many thanks!
<!-- Template for this readme: https://github.com/paperswithcode/releasing-research-code/tree/master -->
