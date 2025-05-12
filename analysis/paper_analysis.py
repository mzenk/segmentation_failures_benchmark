# imports
import colorsys
import json
import pickle
import re
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.markers import MarkerStyle
from utils import load_fd_results_hydra, load_raw_results

from segmentation_failures.evaluation.failure_detection.fd_analysis import (
    ExperimentData,
    compute_fd_scores,
)
from segmentation_failures.evaluation.segmentation.segmentation_metrics import (
    get_metrics_info,
)
from segmentation_failures.utils import GLOBAL_SEEDS
from segmentation_failures.utils.data import get_dataset_dir

dataset_mapper = {
    500: "Brain tumor (2D)",
    503: "Brain tumor",
    514: "Heart (US)",
    510: "Heart (ACDC)",
    511: "Heart",  # M&Ms
    515: "Kidney tumor",
    520: "Covid",
    521: "Prostate",
    540: "OCT (fluids)",
    560: "OCT (layers)",
    531: "Optic cup/disc",
}


pixel_csf_mapper = {
    "baseline": "Single network",
    "mcdropout": "MC-Dropout",
    "deep_ensemble": "Ensemble",
}


image_csf_mapper = {
    "quality_regression": "Quality regression",
    "mahalanobis_gonzalez": "Mahalanobis",
    "vae_mask_only": "VAE (seg)",
    "vae_image_and_mask": "VAE (img + seg)",
}


agg_mapper = {
    "pairwise_dice": "pairwise DSC",
    "pairwise_gen_dice": "pairwise DSC",
    "pairwise_mean_dice": "pairwise DSC",
    "predictive_entropy_mean": "mean PE",
    "predictive_entropy_foreground": "mean foreground PE",
    "predictive_entropy_only_non_boundary": "non-boundary PE",
    "predictive_entropy_patch_min": "patch-based PE",
    "predictive_entropy+heuristic": "RF (simple PE-features)",
    "predictive_entropy+radiomics": "RF (radiomics PE-features)",
}


label_mapper = {
    "aurc": "AURC ↓",
    "norm-aurc": "nAURC ↓",
    "eaurc": "eAURC ↓",
    "dice": "DSC",
    "generalized_dice": "generalized DSC",
    "mean_dice": "mean DSC",
    "spearman": "SC ↓",
    "pearson": "PC ↓",
    "ood_auc": "AUROC (ood) ↑",
    "mean_surface_dice": "mean NSD",
}


color_palette = [
    tuple(int(clr.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
    for clr in sns.color_palette("tab10").as_hex()
]

MAIN_SEG_METRIC = "mean_dice"
PAPER_TEXTWIDTH = 7.23135  # inches
PAPER_TEXTHEIGHT = 9.57262  # inches


def get_figsize(textwidth_factor=1.0, aspect_ratio=None):
    width = PAPER_TEXTWIDTH * textwidth_factor
    if aspect_ratio is None:
        aspect_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
    height = width * aspect_ratio  # figure height in inches
    return width, height


@cache
def get_num_test_cases(dataset_id, root_dir=None):
    if root_dir is None:
        root_dir = "/home/m167k/Datasets/segmentation_failures/nnunet_convention_new"
    # a) count files in the folders <- better
    # b) hard-coded here
    data_root = get_dataset_dir(dataset_id, root_dir)
    return len(list(data_root.glob("labelsTs/*.nii.gz")))


def order_expts(mapped_expt_names, separate_pxl_and_img_csfs=True):
    # use the order implied by the mapper dicts.
    # First pixel confidence scores (order among them by confid name),
    # then image confidence scores (order among them alphabetically)
    def get_order_position(expt_name):
        pixel_confid, image_confid = expt_name.split(" + ")
        pixel_methods = list(pixel_csf_mapper.values())
        img_methods = list(image_csf_mapper.values())
        agg_methods = list(agg_mapper.values())
        # compute an order score, based on the positions in the mapper dicts
        num_per_pixel_confid = len(agg_methods) + len(img_methods)
        if image_confid in agg_methods:
            order_score = pixel_methods.index(
                pixel_confid
            ) * num_per_pixel_confid + agg_methods.index(image_confid)
        else:
            order_score = (
                +pixel_methods.index(pixel_confid) * num_per_pixel_confid
                + len(agg_methods)
                + img_methods.index(image_confid)
            )
            if separate_pxl_and_img_csfs:
                # they will come after all pixel+aggregation methods
                # e.g. single network + mean PE, ensemble + mean PE, single_network + quality regression
                order_score += 1000
        return order_score

    ordered_names = sorted(mapped_expt_names, key=get_order_position)
    if len(ordered_names) != len(mapped_expt_names):
        raise ValueError(f"Unknown experiment name {set(mapped_expt_names) - set(ordered_names)}")
    return ordered_names


def adjust_saturation(rgb, factor):
    """Adjust the saturation of an RGB color."""
    hsl = colorsys.rgb_to_hls(*rgb)
    hsl_adjusted = (hsl[0], min(max(0, hsl[1] * factor), 1), hsl[2])
    return colorsys.hls_to_rgb(*hsl_adjusted)


def save_and_close(
    fig: matplotlib.figure.Figure,
    output_path: Path,
    file_types=None,
    fix_axis_labels=True,
    **kwargs,
):
    if file_types is None:
        file_types = [".png", ".pdf"]
    if output_path.suffix not in file_types:
        file_types.append(output_path.suffix)
    default_kwargs = {
        "dpi": 200,
        "bbox_inches": "tight",
    }
    default_kwargs.update(kwargs)
    if fix_axis_labels:
        for ax in fig.get_axes():
            if ax.get_xlabel():
                ax.set_xlabel(get_nicer_axis_label(ax.get_xlabel()))
            if ax.get_ylabel():
                ax.set_ylabel(get_nicer_axis_label(ax.get_ylabel()))
    for file_type in file_types:
        fig.savefig(output_path.with_suffix(file_type), **default_kwargs)
    plt.close(fig)


def get_nicer_axis_label(label: str):
    if label.lower() in label_mapper:
        return label_mapper[label]
    label = label.replace("_", " ")
    label = label[0].capitalize() + label[1:]
    return label


def map_expt_name(expt_row):
    # drop backbone and segmentation
    if expt_row["pixel_confid"] == "None":
        # should be only mahalanobis
        pixel_confid = pixel_csf_mapper["baseline"]
    else:
        pixel_confid = pixel_csf_mapper[expt_row["pixel_confid"]]
    image_confid = image_csf_mapper.get(expt_row["image_confid"])
    if image_confid is None:
        # aggregation case
        if expt_row["image_confid"] in agg_mapper:
            image_confid = agg_mapper[expt_row["image_confid"]]
        else:
            assert expt_row["image_confid"] == "all_simple"
            image_confid = agg_mapper[expt_row["confid_name"]]
    return f"{pixel_confid} + {image_confid}"


def get_unique_expt_name(expt_name: pd.Series, confid_name: pd.Series):
    if isinstance(expt_name, str):
        expt_name = pd.Series([expt_name])
    if isinstance(confid_name, str):
        confid_name = pd.Series([confid_name])
    assert np.all(expt_name.index == confid_name.index)
    df = expt_name.str.split("-", expand=True)
    df.rename(
        columns={0: "backbone", 1: "segmentation", 2: "pixel_confid", 3: "image_confid"},
        inplace=True,
    )
    df["confid_name"] = confid_name
    return df.apply(map_expt_name, axis=1)


def get_base_color(expt_name, return_idx=True) -> str:
    pxl_csf, image_csf = expt_name.split(" + ")
    img_methods = list(image_csf_mapper.values())
    pxl_methods = list(pixel_csf_mapper.values())
    color_idx = pxl_methods.index(pxl_csf)
    if image_csf in img_methods:
        color_idx = len(pxl_methods) + img_methods.index(image_csf)
    if return_idx:
        return color_idx
    return color_palette[color_idx]


def get_marker(expt_name):
    # depends only on aggregation
    markers = ["o", "o", "o", "s", "p", "v", "D", "P", "X"]
    for i, val in enumerate(agg_mapper.values()):
        if val in expt_name:
            return markers[i]
    return "o"


def get_hue_colors_markers(expt_names, hue_incr=0.1) -> dict:
    base_colors = [get_base_color(x) for x in expt_names]
    # markers depend only on aggregation function
    markers = {x: get_marker(x) for x in expt_names}
    # count how often each color occurs
    shades_counts = defaultdict(int)
    for color_idx in base_colors:
        shades_counts[color_idx] += 1
    # for colors with more than one occurrence, use different shades
    shades_idx = defaultdict(int)
    hue_colors = {}
    for color_idx, key in zip(base_colors, expt_names):
        factor = max(0, 1 + hue_incr * shades_idx[color_idx])
        hue_colors[key] = adjust_saturation(color_palette[color_idx], factor)
        shades_idx[color_idx] += 1
    return hue_colors, markers


def load_results(expt_group_dir, included_datasets=None):
    all_fd_results = []
    all_ood_results = []
    all_configs = {}
    for dataset_dir in expt_group_dir.iterdir():
        if not dataset_dir.name.startswith("Dataset"):
            continue
        dataset_id = int(dataset_dir.name.removeprefix("Dataset"))
        if included_datasets is not None and dataset_id not in included_datasets:
            continue
        fd_results, expt_configs = load_fd_results_hydra(
            dataset_dir / "runs", csv_name="fd_metrics.csv"
        )
        ood_results = None
        if dataset_id not in [503, 515]:
            # no OOD for these datasets
            ood_results, _ = load_fd_results_hydra(
                dataset_dir / "runs", csv_name="ood_metrics.csv"
            )
        if len(expt_configs) == 0:
            logger.warning(f"No experiment runs found for dataset {dataset_id}. Skipping...")
            continue
        # TODO check that expt IDs are consistent, ie the experiment dir (root_dir) in the dataframe is the same
        # add dataset_id to expt_id to avoid ambiguity
        fd_results["expt_id"] = str(dataset_id) + "_" + fd_results["expt_id"].astype(str)
        expt_configs = {f"{dataset_id}_{i}": cfg for i, cfg in expt_configs.items()}

        expt_id_to_name = {i: cfg.expt_name for i, cfg in expt_configs.items()}
        expt_id_to_seed = {i: cfg.seed for i, cfg in expt_configs.items()}
        try:
            expt_id_to_fold = {i: cfg.datamodule.fold for i, cfg in expt_configs.items()}
        except omegaconf.errors.ConfigKeyError:
            # legacy configs
            expt_id_to_fold = {i: cfg.datamodule.hparams.fold for i, cfg in expt_configs.items()}
        fd_results["dataset"] = dataset_id
        fd_results["expt_name"] = fd_results.expt_id.map(expt_id_to_name)
        fd_results["expt_version"] = fd_results.root_dir.str.split("/").str[-1]
        fd_results["seed"] = fd_results.expt_id.map(expt_id_to_seed)
        fd_results["fold"] = fd_results.expt_id.map(expt_id_to_fold)
        if "opt-aurc" in fd_results.columns and "aurc" in fd_results.columns:
            fd_results["eaurc"] = fd_results["aurc"] - fd_results["opt-aurc"]
        all_fd_results.append(fd_results)

        if ood_results is not None and len(ood_results) > 0:
            ood_results["expt_id"] = str(dataset_id) + "_" + ood_results["expt_id"].astype(str)
            ood_results["expt_name"] = ood_results.expt_id.map(expt_id_to_name)
            ood_results["expt_version"] = ood_results.root_dir.str.split("/").str[-1]
            ood_results["dataset"] = dataset_id
            ood_results["seed"] = ood_results.expt_id.map(expt_id_to_seed)
            ood_results["fold"] = ood_results.expt_id.map(expt_id_to_fold)
            all_ood_results.append(ood_results)

        all_configs.update(expt_configs)

    return (
        pd.concat(all_fd_results, ignore_index=True),
        pd.concat(all_ood_results, ignore_index=True) if len(all_ood_results) > 0 else None,
        all_configs,
    )


def filter_expts_for_comparison(fd_results, expt_name_confid_combinations=None):
    # Include these experiments (multiple seeds):
    if expt_name_confid_combinations is None:
        # this is just the default overall comparison
        expt_name_confid_combinations = [
            ("baseline-all_simple", "predictive_entropy_mean"),
            ("deep_ensemble-all_simple", "pairwise_dice"),
            ("mcdropout-all_simple", "pairwise_dice"),
            ("None-mahalanobis", "bottleneck.0.conv2"),
            ("None-mahalanobis", "bottleneck.0.blocks.0.conv2"),  # updated state_dict naming
            ("None-mahalanobis", "model.1.submodule"),
            ("deep_ensemble-quality_regression", MAIN_SEG_METRIC),
            ("deep_ensemble-vae_mask_only", "elbo"),
            # ("None-vae_image_and_mask", "elbo"),
        ]
        if MAIN_SEG_METRIC == "generalized_dice":
            expt_name_confid_combinations += [
                ("deep_ensemble-all_simple", "pairwise_gen_dice"),
                ("mcdropout-all_simple", "pairwise_gen_dice"),
            ]
        elif MAIN_SEG_METRIC == "mean_dice":
            expt_name_confid_combinations += [
                ("deep_ensemble-all_simple", "pairwise_mean_dice"),
                ("mcdropout-all_simple", "pairwise_mean_dice"),
            ]
    filtered_df = []
    for expt_name, confid_name in expt_name_confid_combinations:
        if confid_name == "mean_dice":
            # TODO this is a workaround because I changed some confidence names
            # use the dice score for the only fg class for these datasets
            filtered_df.append(
                fd_results[
                    (fd_results.dataset.isin([500, 520, 521]))
                    & (fd_results.expt_name.str.contains(expt_name, regex=False))
                    & (fd_results.confid_name.str.contains("dice_0", regex=False))
                ]
            )
        filtered_df.append(
            fd_results[
                (fd_results.expt_name.str.contains(expt_name, regex=False))
                & (fd_results.confid_name.str.contains(confid_name, regex=False))
            ]
        )
    return pd.concat(filtered_df)


def plot_confidence_vs_metric(
    results_df: pd.DataFrame,
    metric: str,
    confid_name: str,
    title=None,
    plot_regression_line: bool = False,
    normalize_confid=False,
    show_diagonal_line=False,
    **plot_kwargs,
):
    fig, ax = plt.subplots()
    # this is how the columns are named
    metric_col = f"metric_{metric}"
    confid_col = f"confid_{confid_name}"
    # curr_ax.set_aspect("equal")
    plot_data = results_df.copy()
    # normalize confidence (min/max)
    if normalize_confid:
        normalized_confids = plot_data.groupby("expt_id", group_keys=False)[confid_col].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        assert np.all(normalized_confids.index == plot_data.index)
        plot_data[confid_col] = normalized_confids
    sns.scatterplot(
        x=confid_col,
        y=metric_col,
        hue="domain",
        data=plot_data,
        ax=ax,
        alpha=0.8,
        **plot_kwargs,
    )
    if show_diagonal_line:
        ax.plot([0, 1], [0, 1], linestyle=":", color="black", alpha=0.3)
    sns.move_legend(
        ax,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        ncol=1,
        title="Domain",
        frameon=True,
    )
    if plot_regression_line:
        # plot the regression line
        sns.regplot(
            x=confid_col,
            y=metric_col,
            data=plot_data,
            ax=ax,
            scatter=False,
            color="black",
            line_kws={"alpha": 0.7},
        )
    # curr_ax.set_ylabel(metric)
    ax.set_ylabel(metric)
    ax.set_xlabel("Confidence")
    if title is not None:
        ax.set_title(title)
    return fig


# NOTE adapted from standard plots. lazy...
def plot_class_seg_metrics_across_domains(
    df: pd.DataFrame, metric: str, included_domains: list = None, ax=None
):
    plot_data = df.copy()
    # match all columns that follow the pattern metric_classXX_metric, where XX are two digits
    # class_metrics = [c for c in plot_data.columns if c.endswith(metric) and "metric_class" in c]
    pattern = re.compile(r"metric_class\d\d_" + metric)
    class_metrics = [c for c in plot_data.columns if pattern.match(c)]
    if len(class_metrics) == 0:
        plot_data.rename(columns={f"metric_{metric}": f"metric_class00_{metric}"}, inplace=True)
        class_metrics = [
            c for c in plot_data.columns if c.endswith(metric) and "metric_class" in c
        ]

    plot_data = plot_data.melt(
        id_vars=["domain", "case_id"],
        value_vars=class_metrics,
        var_name="class",
        value_name=metric,
    )
    if included_domains is not None:
        plot_data = plot_data[plot_data.domain.isin(included_domains)]
    shift_order = (
        plot_data.groupby("domain")[metric].median().sort_values(ascending=False).index.tolist()
    )
    if included_domains is not None:
        shift_order = included_domains
    if ax is None:
        fig, ax = plt.subplots()
    sns.boxplot(
        data=plot_data,
        x="domain",
        y=metric,
        hue="class",
        order=shift_order,
        fliersize=0,
        ax=ax,
        palette="gray",
        fill=False,
    )
    sns.stripplot(
        data=plot_data,
        x="domain",
        y=metric,
        hue="class",
        jitter=0.1,
        dodge=True,
        order=shift_order,
        alpha=0.5,
        ax=ax,
        palette="rocket",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[len(handles) // 2 :], labels[len(labels) // 2 :])
    if len(class_metrics) == 1:
        # disable legend
        ax.legend().set_visible(False)
    return ax


# TODO maybe merge with the _with_ranking variant?
def plot_fd_comparison_across_datasets(
    df, seg_metric, fd_metric, change_markers=False, show_rand_opt=False, **strip_kwargs
):
    plot_data = df.copy()
    plot_data = plot_data[plot_data.metric == seg_metric]
    assert plot_data.domain.nunique() == 1
    plot_data = plot_data.sort_values("expt_name")
    # drop duplicates (I re-ran some experiments)
    plot_data = plot_data.drop_duplicates(
        subset=[x for x in plot_data.columns if x not in ["expt_version", "expt_id", "root_dir"]]
    )
    # make the expt_name and dataset description nicer
    plot_data["legend_name"] = get_unique_expt_name(plot_data.expt_name, plot_data.confid_name)
    plot_data["dataset"] = plot_data.dataset.map(dataset_mapper)
    expt_order = order_expts(plot_data["legend_name"].unique().tolist())
    ds_order = [x for x in dataset_mapper.values() if x in plot_data.dataset.unique()]
    colors, markers = get_hue_colors_markers(expt_order, hue_incr=0.12 * (not change_markers))
    # check for more subtle duplicates
    for group, group_df in plot_data.groupby(["dataset", "legend_name"]):
        num_seeds = group_df.seed.nunique()
        num_folds = group_df.fold.nunique()
        if len(group_df) != 5:
            logger.warning(
                f"Found {len(group_df)} results ({num_seeds} seeds and {num_folds} folds) for {group} but expected 5."
            )
    # either vary hue or markers...?
    if not change_markers:
        markers = {k: "o" for k in markers}
    fig, ax = plt.subplots(figsize=get_figsize(), layout="constrained")
    for expt_name, m in markers.items():
        sns.stripplot(
            data=plot_data[plot_data.legend_name == expt_name],
            x="dataset",
            y=fd_metric,
            hue="legend_name",
            ax=ax,
            dodge=True,
            order=ds_order,
            palette=colors,
            marker=m,
            hue_order=expt_order,
            **strip_kwargs,
        )
    # remove duplicates from legend and order by expt_order
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # add vertical separators between datasets
    for i in range(1, len(plot_data.dataset.unique())):
        ax.axvline(i - 0.5, color="gray", linestyle=":", alpha=0.5)
    if fd_metric == "norm-aurc":
        ax.axhline(y=1, color="black", linestyle="--")
        ax.set_ylim(bottom=-0.05)
    if fd_metric.endswith("auc"):
        ax.axhline(y=0.5, color="black", linestyle="--")
        ax.set_ylim(top=1.05)
    if (
        fd_metric == "aurc"
        and show_rand_opt
        and {"rand-aurc", "opt-aurc"}.issubset(plot_data.columns)
    ):
        sns.stripplot(
            data=plot_data,
            x="dataset",
            y="rand-aurc",
            hue="legend_name",
            ax=ax,
            dodge=True,
            order=ds_order,
            palette=["gray"] * len(expt_order),
            hue_order=expt_order,
            marker="$-$",
            alpha=0.2,
            jitter=0,
        )
        sns.stripplot(
            data=plot_data,
            x="dataset",
            y="opt-aurc",
            hue="legend_name",
            ax=ax,
            dodge=True,
            order=ds_order,
            palette=["gray"] * len(expt_order),
            hue_order=expt_order,
            marker="$-$",
            alpha=0.2,
            jitter=0,
        )
    # remove duplicates from legend and order by expt_order
    ax.legend(
        [
            plt.Line2D([], [], marker=markers[x], color=by_label[x]._color, linestyle="")
            for x in expt_order
        ],
        expt_order,
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, 1),
        ncol=1,
        title=None,
        frameon=False,
    )
    return fig


def plot_fd_comparison_across_datasets_multi(
    df, seg_metric, fd_metric, change_markers=False, show_rand_opt=False, **strip_kwargs
):
    # fd_metric can be a list
    if isinstance(fd_metric, str):
        return plot_fd_comparison_across_datasets(
            df, seg_metric, fd_metric, change_markers, show_rand_opt, **strip_kwargs
        )
    plot_data = df.copy()
    plot_data = plot_data[plot_data.metric == seg_metric]
    assert plot_data.domain.nunique() == 1
    plot_data = plot_data.sort_values("expt_name")
    # drop duplicates (I re-ran some experiments)
    plot_data = plot_data.drop_duplicates(
        subset=[x for x in plot_data.columns if x not in ["expt_version", "expt_id", "root_dir"]]
    )
    # make the expt_name and dataset description nicer
    plot_data["legend_name"] = get_unique_expt_name(plot_data.expt_name, plot_data.confid_name)
    plot_data["dataset"] = plot_data.dataset.map(dataset_mapper)
    expt_order = order_expts(plot_data["legend_name"].unique().tolist())
    ds_order = [x for x in dataset_mapper.values() if x in plot_data.dataset.unique()]
    colors, markers = get_hue_colors_markers(expt_order, hue_incr=0.12 * (not change_markers))
    # check for more subtle duplicates
    for group, group_df in plot_data.groupby(["dataset", "legend_name"]):
        num_seeds = group_df.seed.nunique()
        num_folds = group_df.fold.nunique()
        if len(group_df) != 5:
            logger.warning(
                f"Found {len(group_df)} results ({num_seeds} seeds and {num_folds} folds) for {group} but expected 5."
            )
    # either vary hue or markers...?
    if not change_markers:
        markers = {k: "o" for k in markers}
    fig, axes = plt.subplots(
        len(fd_metric), 1, figsize=(PAPER_TEXTWIDTH, PAPER_TEXTHEIGHT * 0.95), layout="constrained"
    )
    for idx, fdm in enumerate(fd_metric):
        ax = axes[idx]
        for expt_name, m in markers.items():
            sns.stripplot(
                data=plot_data[plot_data.legend_name == expt_name],
                x="dataset",
                y=fdm,
                hue="legend_name",
                ax=ax,
                dodge=True,
                order=ds_order,
                palette=colors,
                marker=m,
                hue_order=expt_order,
                **strip_kwargs,
            )
        # remove duplicates from legend and order by expt_order
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # add vertical separators between datasets
        for i in range(1, len(plot_data.dataset.unique())):
            ax.axvline(i - 0.5, color="gray", linestyle=":", alpha=0.5)
        if fdm == "norm-aurc":
            ax.axhline(y=1, color="black", linestyle="--")
            ax.set_ylim(bottom=-0.05)
        if fdm.endswith("auc"):
            ax.axhline(y=0.5, color="black", linestyle="--")
            ax.set_ylim(top=1.05)
        if (
            fdm == "aurc"
            and show_rand_opt
            and {"rand-aurc", "opt-aurc"}.issubset(plot_data.columns)
        ):
            sns.stripplot(
                data=plot_data,
                x="dataset",
                y="rand-aurc",
                hue="legend_name",
                ax=ax,
                dodge=True,
                order=ds_order,
                palette=["gray"] * len(expt_order),
                hue_order=expt_order,
                marker="$-$",
                alpha=0.2,
                jitter=0,
            )
            sns.stripplot(
                data=plot_data,
                x="dataset",
                y="opt-aurc",
                hue="legend_name",
                ax=ax,
                dodge=True,
                order=ds_order,
                palette=["gray"] * len(expt_order),
                hue_order=expt_order,
                marker="$-$",
                alpha=0.2,
                jitter=0,
            )
        # remove duplicates from legend and order by expt_order
        if idx == 0:
            ax.legend(
                [
                    plt.Line2D([], [], marker=markers[x], color=by_label[x]._color, linestyle="")
                    for x in expt_order
                ],
                expt_order,
                # loc="lower center",
                # bbox_to_anchor=(0.5, 1),
                loc="upper right",
                bbox_to_anchor=(1, 1),
                ncols=2,
                title=None,
                # frameon=False,
            )
        else:
            ax.legend().set_visible(False)
        if idx != len(fd_metric) - 1:
            ax.set_xlabel("")
    return fig


def plot_fd_comparison_across_datasets_with_ranking(
    df, seg_metric, fd_metric, show_rand_opt=False, **strip_kwargs
):
    plot_data = df.copy()
    plot_data = plot_data[plot_data.metric == seg_metric]
    assert plot_data.domain.nunique() == 1
    plot_data = plot_data.sort_values("expt_name")
    # drop duplicates (I re-ran some experiments)
    plot_data = plot_data.drop_duplicates(
        subset=[x for x in plot_data.columns if x not in ["expt_version", "expt_id", "root_dir"]]
    )
    # make the expt_name and dataset description nicer
    plot_data["legend_name"] = get_unique_expt_name(plot_data.expt_name, plot_data.confid_name)
    plot_data["dataset"] = plot_data.dataset.map(dataset_mapper)
    expt_order = order_expts(plot_data["legend_name"].unique().tolist())
    ds_order = [x for x in dataset_mapper.values() if x in plot_data.dataset.unique()]
    colors, markers = get_hue_colors_markers(expt_order, hue_incr=0.12)
    # check for more subtle duplicates
    for group, group_df in plot_data.groupby(["dataset", "legend_name"]):
        num_seeds = group_df.seed.nunique()
        num_folds = group_df.fold.nunique()
        if len(group_df) != 5:
            logger.warning(
                f"Found {len(group_df)} results ({num_seeds} seeds and {num_folds} folds) for {group} but expected 5."
            )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=get_figsize(),
        layout="constrained",
        gridspec_kw={"height_ratios": [1, 3.5]},
        sharex=True,
    )
    # upper plot:
    ax = axes[0]
    # compute ranking based on mean fd_metric
    # adapted from ranking plot
    group_df = plot_data.groupby(["dataset", "legend_name"])
    for _, df in group_df:
        if "deep_ensemble" in df.expt_name.unique()[0]:
            assert (
                df.seed.nunique() <= 3
            ), f"Found {df.seed.nunique()} seeds for {df.expt_name.unique()[0]}"
        else:
            assert (
                df.seed.nunique() <= 5
            ), f"Found {df.seed.nunique()} seeds for {df.expt_name.unique()[0]}"
    avg_metric = group_df[fd_metric].agg("mean")
    higher_better = True if fd_metric == "norm-aurc" else False
    # for correlation metrics, negative values are better (risk vs confidence)
    if fd_metric not in ["aurc", "eaurc", "norm-aurc", "spearman", "pearson"]:
        raise ValueError(f"Unknown metric {fd_metric}")
    avg_metric_rank = avg_metric.groupby("dataset").rank(ascending=not higher_better).reset_index()
    sns.pointplot(
        data=avg_metric_rank,
        x="dataset",
        y=fd_metric,
        hue="legend_name",
        dodge=False,
        linestyles="None",
        markers=MarkerStyle("s").scaled(10, 1),
        ax=ax,
        palette=colors,
        order=ds_order,
        hue_order=expt_order,
    )
    ax.set_ylabel("Ranking")
    ax.set_yticks(range(1, len(expt_order) + 1))
    ax.set_ylim(1 - 0.4, len(expt_order) + 0.4)
    # Get the handles and labels from the plot
    handles, labels = ax.get_legend_handles_labels()

    # # Create custom handles
    # custom_handles = [
    #     plt.Line2D([], [], marker="o", color=h._color, linestyle="") for h in handles
    # ]

    # Create the legend with the custom handles
    # ax.legend(
    #     custom_handles,
    #     labels,
    #     loc="lower center",
    #     ncol=2,
    #     bbox_to_anchor=(0.5, 1),
    #     title=None,
    #     frameon=False,
    # )
    ax.legend().set_visible(False)
    # lower plot:
    ax = axes[1]
    sns.stripplot(
        data=plot_data,
        x="dataset",
        y=fd_metric,
        hue="legend_name",
        ax=ax,
        dodge=True,
        order=ds_order,
        palette=colors,
        hue_order=expt_order,
        **strip_kwargs,
    )
    # ax.legend().set_visible(False)
    # Get the handles and labels from the plot
    handles, labels = ax.get_legend_handles_labels()
    # add vertical separators between datasets
    for i in range(1, len(plot_data.dataset.unique())):
        ax.axvline(i - 0.5, color="gray", linestyle=":", alpha=0.5)
    if fd_metric == "norm-aurc":
        ax.axhline(y=1, color="black", linestyle="--")
        ax.set_ylim(bottom=-0.05)
    if fd_metric.endswith("auc"):
        ax.axhline(y=0.5, color="black", linestyle="--")
        ax.set_ylim(top=1.05)
    if (
        fd_metric == "aurc"
        and show_rand_opt
        and {"rand-aurc", "opt-aurc"}.issubset(plot_data.columns)
    ):
        sns.stripplot(
            data=plot_data,
            x="dataset",
            y="rand-aurc",
            hue="legend_name",
            ax=ax,
            dodge=True,
            order=ds_order,
            palette=["gray"] * len(expt_order),
            hue_order=expt_order,
            marker="$-$",
            alpha=0.2,
            jitter=0,
        )
        sns.stripplot(
            data=plot_data,
            x="dataset",
            y="opt-aurc",
            hue="legend_name",
            ax=ax,
            dodge=True,
            order=ds_order,
            palette=["gray"] * len(expt_order),
            hue_order=expt_order,
            marker="$-$",
            alpha=0.2,
            jitter=0,
        )
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[: len(expt_order)]
        labels = labels[: len(expt_order)]
        # ax.legend().set_visible(False)
    ax.legend(
        handles,
        labels,
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0, 1),
        title=None,
        frameon=True,
    )
    return fig


def plot_ood_comparison_across_datasets(df, ood_metric, domain="all_ood_"):
    plot_data = df.copy()
    plot_data = plot_data[plot_data.domain == domain]
    # drop duplicates (I re-ran some experiments)
    plot_data = plot_data.drop_duplicates(
        subset=[x for x in plot_data.columns if x not in ["expt_version", "expt_id", "root_dir"]]
    )
    # make the expt_name and dataset description nicer
    plot_data["legend_name"] = get_unique_expt_name(plot_data.expt_name, plot_data.confid_name)
    plot_data["dataset"] = plot_data.dataset.map(dataset_mapper)
    expt_order = order_expts(plot_data["legend_name"].unique().tolist())
    colors, markers = get_hue_colors_markers(expt_order)
    # check for more subtle duplicates
    for group, group_df in plot_data.groupby(["dataset", "legend_name"]):
        num_seeds = group_df.seed.nunique()
        num_folds = group_df.fold.nunique()
        if len(group_df) != 5:
            logger.warning(
                f"Found {len(group_df)} results ({num_seeds} seeds and {num_folds} folds) for {group} but expected 5."
            )
    fig, ax = plt.subplots(figsize=get_figsize(textwidth_factor=0.8))
    sns.stripplot(
        data=plot_data,
        x="dataset",
        y=ood_metric,
        hue="legend_name",
        ax=ax,
        dodge=True,
        order=[x for x in dataset_mapper.values() if x in plot_data.dataset.unique()],
        palette=colors,
        hue_order=expt_order,
    )
    sns.move_legend(
        ax,
        "upper left",
        ncols=2,
        bbox_to_anchor=(0, 1),
        title=None,
    )
    # add vertical separators between datasets
    for i in range(1, len(plot_data.dataset.unique())):
        ax.axvline(i - 0.5, color="gray", linestyle=":", alpha=0.5)
    if ood_metric.endswith("auc"):
        ax.axhline(y=0.5, color="black", linestyle="--")
        ax.set_ylim(top=1.05)
    fig.tight_layout()
    return fig


def plot_risk_coverage_curves(fd_results, risk_metric):
    # Each row in fd_results is a single experiment and we plot the risk-coverage curve for each of them
    # We need to load the curve npz files and then plot them
    assert fd_results.dataset.nunique() == 1
    plot_data = filter_expts_for_comparison(fd_results)
    assert plot_data.domain.nunique() == 1
    plot_data = plot_data[plot_data.metric == risk_metric]
    fig, ax = plt.subplots()
    # make the expt_name and dataset description nicer
    plot_data["legend_name"] = get_unique_expt_name(plot_data.expt_name, plot_data.confid_name)
    expt_order = order_expts(plot_data.legend_name.unique().tolist())
    colors, markers = get_hue_colors_markers(expt_order)
    num_runs_per_expt = plot_data.groupby("legend_name").size()
    num_seeds_per_expt = plot_data.groupby("legend_name").seed.nunique()
    num_folds_per_expt = plot_data.groupby("legend_name").fold.nunique()
    if not np.all(num_runs_per_expt == num_seeds_per_expt):
        logger.warning(
            "Found different number of runs per experiment. "
            "Maybe some seeds were re-run? Consider dropping them manually."
        )
    for _, row in plot_data.iterrows():
        curve_file = row.file_risk_coverage_curve
        if not Path(curve_file).is_absolute():
            curve_file = f"{row.root_dir}/analysis/{curve_file}"
        curr_curve = np.load(curve_file)
        num_seeds = num_seeds_per_expt.loc[row.legend_name]
        num_folds = num_folds_per_expt.loc[row.legend_name]
        plt.plot(
            curr_curve["coverage"],
            curr_curve["risk"],
            "-",
            label=row.legend_name,
            color=colors[row.legend_name],
            alpha=0.5 if max(num_seeds, num_folds) > 1 else 1,
        )
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk")
    ax.legend()
    # remove duplicates from legend and order by expt_order
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend([by_label[x] for x in expt_order], expt_order)
    sns.move_legend(
        ax,
        "lower center",
        ncols=2,
        bbox_to_anchor=(0.5, 1),
        ncol=1,
        title=None,
        frameon=False,
    )
    return fig


def make_table_dice_vs_surf_dist(
    fd_results, fd_metric, output_file, surface_distance_metric="mean_surface_dice"
):
    plot_data = filter_expts_for_comparison(fd_results)
    plot_data = plot_data[plot_data.metric.isin([MAIN_SEG_METRIC, surface_distance_metric])]
    assert plot_data.domain.nunique() == 1
    plot_data = plot_data.drop_duplicates(
        subset=[x for x in plot_data.columns if x not in ["expt_version", "expt_id", "root_dir"]]
    )
    # check for more subtle duplicates
    for group, group_df in plot_data.groupby(["expt_name", "dataset", "confid_name", "metric"]):
        num_seeds = group_df.seed.nunique()
        num_folds = group_df.fold.nunique()
        if len(group_df) > max(num_folds, num_seeds):
            logger.warning(
                f"Found {len(group_df)} results for {group}."
                "Maybe some seeds were re-run? Consider dropping them manually."
            )
    # make the expt_name and dataset description nicer
    plot_data["nice_expt_name"] = get_unique_expt_name(plot_data.expt_name, plot_data.confid_name)
    plot_data["dataset"] = plot_data.dataset.map(dataset_mapper)
    plot_data["metric"] = plot_data.metric.map(label_mapper)
    assert plot_data.metric.nunique() == 2

    plot_data = plot_data.pivot(
        index=["nice_expt_name", "expt_id"], columns=["dataset", "metric"], values=fd_metric
    )
    # compute the mean and std across expt_id (seeds) for each column
    plot_data = plot_data.groupby(level=0, axis=0).agg(["mean", "std"])
    expt_order = order_expts(plot_data.index.get_level_values(0).unique().tolist())
    plot_data = plot_data.reindex(expt_order, axis=0, level=0)

    idx = pd.IndexSlice
    mean_slice_ = idx[:, idx[:, :, "mean"]]
    std_slice_ = idx[:, idx[:, :, "std"]]
    dice_slice_ = idx[:, idx[:, label_mapper[MAIN_SEG_METRIC], :]]
    sdist_slice_ = idx[:, idx[:, label_mapper[surface_distance_metric], :]]

    # Table style
    plot_data *= 100
    s = plot_data.style.format(precision=1, na_rep="n/a", subset=dice_slice_)
    s.format(precision=1, na_rep="n/a", subset=sdist_slice_)
    s.set_properties(subset=std_slice_, **{"color": "gray"})
    # either highlight min or color map
    # s.highlight_min(axis=0, subset=mean_slice_, props="font-weight:bold")
    # cm = sns.light_palette("green", as_cmap=True)
    cm = sns.color_palette("flare", as_cmap=True)
    s.background_gradient(cmap=cm, axis=0, subset=mean_slice_)
    s.to_html(
        output_file,
        # column_format="rrrrr",
        hrules=True,
        multirow_align="t",
        multicol_align="r",
    )


def make_table_big_comparison(fd_results, fd_metric, seg_metric, output_file, included_expts=None):
    output_file = Path(output_file)
    if included_expts is None:
        included_expts = []
        for group_name, _ in fd_results.groupby(["expt_name", "confid_name"]):
            expt_name, confid_name = group_name
            if re.search(r"mutual_information|maxsoftmax|dice_\d", confid_name) is not None:
                continue
            # not sure about these cases
            if seg_metric == "generalized_dice" and (
                "mean_dice" in confid_name or "pairwise_mean_dice" in confid_name
            ):
                continue
            elif seg_metric == "mean_dice" and (
                "generalized_dice" in confid_name or "pairwise_gen_dice" in confid_name
            ):
                continue
            if "baseline-vae" in expt_name or "mcdropout-vae" in expt_name:
                # not available for all datasets
                continue
            included_expts.append((expt_name, confid_name))
        # legacy from naming changes when switching to mean_dice
        included_expts += [
            ("baseline-predictive_entropy+heuristic", MAIN_SEG_METRIC),
            ("baseline-predictive_entropy+radiomics", MAIN_SEG_METRIC),
            ("baseline-quality_regression", MAIN_SEG_METRIC),
            ("deep_ensemble-predictive_entropy+heuristic", MAIN_SEG_METRIC),
            ("deep_ensemble-predictive_entropy+radiomics", MAIN_SEG_METRIC),
            ("deep_ensemble-quality_regression", MAIN_SEG_METRIC),
        ]
    plot_data = filter_expts_for_comparison(
        fd_results, expt_name_confid_combinations=included_expts
    )
    plot_data = plot_data[plot_data.metric.isin([seg_metric])]
    assert plot_data.domain.nunique() == 1
    plot_data = plot_data.drop_duplicates(
        subset=[x for x in plot_data.columns if x not in ["expt_version", "expt_id", "root_dir"]]
    )
    # make the expt_name and dataset description nicer
    plot_data["Method"] = get_unique_expt_name(plot_data.expt_name, plot_data.confid_name)
    plot_data["dataset"] = plot_data.dataset.map(dataset_mapper)
    plot_data["metric"] = plot_data.metric.map(label_mapper)
    assert plot_data.metric.nunique() == 1
    # check for more subtle duplicates
    for group, group_df in plot_data.groupby(["dataset", "Method"]):
        num_seeds = group_df.seed.nunique()
        num_folds = group_df.fold.nunique()
        if len(group_df) != 5:
            raise ValueError(
                f"Found {len(group_df)} results ({num_seeds} seeds and {num_folds} folds) for {group} but expected 5."
            )
    plot_data = plot_data.pivot(index=["Method", "expt_id"], columns=["dataset"], values=fd_metric)
    plot_data.columns.name = "Dataset"
    # compute the mean and std across expt_id (seeds) for each column
    plot_data = plot_data.groupby(level=0, axis=0).agg(["mean", "std"])
    expt_order = order_expts(
        plot_data.index.get_level_values(0).unique().tolist(), separate_pxl_and_img_csfs=False
    )
    plot_data = plot_data.reindex(expt_order, axis=0, level=0)
    plot_data = plot_data.rename_axis(columns=["Dataset", label_mapper[fd_metric]], index=None)
    idx = pd.IndexSlice
    mean_slice_ = idx[:, idx[:, "mean"]]
    std_slice_ = idx[:, idx[:, "std"]]

    # Table style
    plot_data *= 100
    s = plot_data.style.format(precision=1, na_rep="n/a")
    s.set_properties(subset=std_slice_, **{"color": "gray"})
    # either highlight min or color map
    cm = sns.color_palette("YlGn_r", as_cmap=True)
    s.background_gradient(cmap=cm, axis=0, subset=mean_slice_)
    # s.hide(axis=1, level=1)
    s.to_html(
        output_file.with_suffix(".html"),
        hrules=True,
        multicol_align="c",
    )
    s.to_latex(
        output_file,
        caption="Selected stock correlation and simple statistics.",
        column_format="l" + "rl" * 6,
        clines="skip-last;data",
        convert_css=True,
        position_float="centering",
        multicol_align="c",
        hrules=True,
    )
    return s


def make_table_big_comparison_multiple_metrics(
    fd_results, fd_metric_list, seg_metric, output_file
):
    output_file = Path(output_file)
    included_expts = []
    for group_name, _ in fd_results.groupby(["expt_name", "confid_name"]):
        expt_name, confid_name = group_name
        if re.search(r"mutual_information|maxsoftmax|dice_\d", confid_name) is not None:
            continue
        # not sure about these cases
        if seg_metric == "generalized_dice" and (
            "mean_dice" in confid_name or "pairwise_mean_dice" in confid_name
        ):
            continue
        elif seg_metric == "mean_dice" and (
            "generalized_dice" in confid_name or "pairwise_gen_dice" in confid_name
        ):
            continue
        if "baseline-vae" in expt_name or "mcdropout-vae" in expt_name:
            # not available for all datasets
            continue
        included_expts.append((expt_name, confid_name))
    # legacy from naming changes when switching to mean_dice
    included_expts += [
        ("baseline-predictive_entropy+heuristic", MAIN_SEG_METRIC),
        ("baseline-predictive_entropy+radiomics", MAIN_SEG_METRIC),
        ("baseline-quality_regression", MAIN_SEG_METRIC),
        ("deep_ensemble-predictive_entropy+heuristic", MAIN_SEG_METRIC),
        ("deep_ensemble-predictive_entropy+radiomics", MAIN_SEG_METRIC),
        ("deep_ensemble-quality_regression", MAIN_SEG_METRIC),
    ]
    plot_data = filter_expts_for_comparison(
        fd_results, expt_name_confid_combinations=included_expts
    )
    plot_data = plot_data[plot_data.metric.isin([seg_metric])]
    assert plot_data.domain.nunique() == 1
    plot_data = plot_data.drop_duplicates(
        subset=[x for x in plot_data.columns if x not in ["expt_version", "expt_id", "root_dir"]]
    )
    # make the expt_name and dataset description nicer
    plot_data["Method"] = get_unique_expt_name(plot_data.expt_name, plot_data.confid_name)
    plot_data["dataset"] = plot_data.dataset.map(dataset_mapper)
    plot_data["metric"] = plot_data.metric.map(label_mapper)
    assert plot_data.metric.nunique() == 1
    # check for more subtle duplicates
    for group, group_df in plot_data.groupby(["dataset", "Method"]):
        num_seeds = group_df.seed.nunique()
        num_folds = group_df.fold.nunique()
        if len(group_df) != 5:
            raise ValueError(
                f"Found {len(group_df)} results ({num_seeds} seeds and {num_folds} folds) for {group} but expected 5."
            )
    # plot_data = plot_data.pivot(index=["Method", "fold"], columns=["dataset"], values=fd_metric_list)
    plot_data = plot_data.set_index(["Method", "fold", "dataset"])
    plot_data = (
        plot_data[fd_metric_list].unstack("dataset").swaplevel(0, 1, axis=1).sort_index(axis=1)
    )
    plot_data = plot_data.rename(
        {k: v.removesuffix(" ↓").removesuffix(" ↑") for k, v in label_mapper.items()},
        axis=1,
        level=1,
    )
    # plot_data.columns.name = "Dataset"
    # compute the mean across expt_id (seeds/folds) for each column
    plot_data = plot_data.groupby(level="Method").agg("mean")
    expt_order = order_expts(
        plot_data.index.get_level_values(0).unique().tolist(), separate_pxl_and_img_csfs=False
    )
    plot_data = plot_data.reindex(expt_order, axis=0, level=0)

    def quick_renamer(x: str):
        x = x.replace("Single network", "SN")
        x = x.replace("Ensemble", "DE")
        x = x.replace("MC-Dropout", "MCD")
        return x

    plot_data = plot_data.rename(index=quick_renamer)

    # drop 2D dataset for space reasons
    plot_data = plot_data.drop(columns="Brain tumor (2D)")

    # Table style
    s = plot_data.style.format(precision=3, na_rep="n/a")
    # s = plot_data.style.format(lambda x: f"{x:.3f}".lstrip('0') if isinstance(x, float) else x, na_rep="n/a")
    # either highlight min or color map
    # s.highlight_min(axis=0, subset=mean_slice_, props="font-weight:bold")
    # cm = sns.light_palette("green", as_cmap=True)
    cm = sns.color_palette("YlGn_r", as_cmap=True)
    s.background_gradient(cmap=cm, axis=0)
    # s.background_gradient(cmap=cm, axis=None, subset=mean_slice_)
    # s.hide(axis=1, level=1)
    s.to_html(
        output_file.with_suffix(".html"),
        hrules=True,
        multicol_align="c",
    )
    num_datasets = plot_data.columns.get_level_values(0).nunique()
    s.to_latex(
        output_file,
        caption="Selected stock correlation and simple statistics.",
        column_format="l" + ("|" + "c" * len(fd_metric_list)) * num_datasets,
        clines="skip-last;data",
        convert_css=True,
        position_float="centering",
        multicol_align="c",
        hrules=True,
    )


def make_plots_confid_vs_metric(fd_results, expt_configs, metric, output_dir):
    assert fd_results.dataset.nunique() == 1
    dataset_id = fd_results.dataset.unique()[0]
    plot_data = filter_expts_for_comparison(fd_results)
    plot_data = plot_data[plot_data.metric == metric]
    plot_data = plot_data[plot_data.seed == GLOBAL_SEEDS[0]]
    # First I need to load the raw results for these experiments (segmentation scores and confidences)
    for expt_id, confid_name in plot_data.groupby(["expt_id", "confid_name"]).groups.keys():
        expt_name = plot_data[plot_data.expt_id == expt_id].expt_name.unique()[0]
        curr_results = load_raw_results(expt_configs[expt_id].paths.output_dir)
        curr_results["expt_id"] = expt_id
        included_domains = [
            d for d in curr_results.domain.unique() if d not in ["all_", "all_ood_"]
        ]
        curr_results = curr_results[curr_results.domain.isin(included_domains)]
        nice_expt_name = get_unique_expt_name(expt_name, confid_name).iloc[0]
        fig = plot_confidence_vs_metric(
            curr_results,
            metric,
            confid_name,
            title=nice_expt_name,
            plot_regression_line=False,
            show_diagonal_line="pairwise" in nice_expt_name,
        )
        save_and_close(fig, output_dir / f"dataset{dataset_id}_{nice_expt_name}_{metric}.png")


def make_figure_overall_ranking(fd_results, expt_configs, seg_metric, fd_metric):
    plot_data = filter_expts_for_comparison(fd_results)
    # still need to select the risk metric
    plot_data = plot_data[plot_data.metric == seg_metric]
    # rank the expt_ids by their mean metric (across seeds)
    group_df = plot_data.groupby(["dataset", "expt_name", "confid_name"])
    for _, df in group_df:
        if "deep_ensemble" in df.expt_name.unique()[0]:
            assert (
                df.seed.nunique() <= 3
            ), f"Found {df.seed.nunique()} seeds for {df.expt_name.unique()[0]}"
        else:
            assert (
                df.seed.nunique() <= 5
            ), f"Found {df.seed.nunique()} seeds for {df.expt_name.unique()[0]}"
    avg_metric = group_df[fd_metric].agg("mean")
    higher_better = True if fd_metric == "norm-aurc" else False
    # for correlation metrics, negative values are better (risk vs confidence)
    if fd_metric not in ["aurc", "eaurc", "norm-aurc", "spearman", "pearson"]:
        raise ValueError(f"Unknown metric {fd_metric}")
    avg_metric_rank = avg_metric.groupby("dataset").rank(ascending=not higher_better).reset_index()
    # name niceties
    avg_metric_rank["nice_expt_name"] = get_unique_expt_name(
        avg_metric_rank.expt_name, avg_metric_rank.confid_name
    )
    avg_metric_rank["nice_dataset"] = avg_metric_rank.dataset.map(dataset_mapper)
    hue_order = order_expts(avg_metric_rank.nice_expt_name.unique().tolist())
    hue_colors, markers = get_hue_colors_markers(hue_order)
    fig, ax = plt.subplots()
    sns.pointplot(
        data=avg_metric_rank,
        x="nice_dataset",
        y=fd_metric,
        hue="nice_expt_name",
        dodge=False,
        linestyles=":",
        ax=ax,
        palette=hue_colors,
        order=[x for x in dataset_mapper.values() if x in avg_metric_rank.nice_dataset.unique()],
        hue_order=hue_order,
    )
    sns.move_legend(
        ax,
        "lower center",
        ncols=2,
        bbox_to_anchor=(0.5, 1),
        ncol=1,
        title=None,
        frameon=False,
    )
    ax.set_ylabel("Ranking")
    ax.set_xlabel("Dataset")
    return fig


def make_figure_overall_comparison(
    fd_results, expt_configs, seg_metric, fd_metric, include_ranking=False
):
    plot_data = filter_expts_for_comparison(fd_results)
    if include_ranking:
        return plot_fd_comparison_across_datasets_with_ranking(
            plot_data, seg_metric, fd_metric, show_rand_opt=True, alpha=0.8
        )
    if isinstance(fd_metric, str):
        return plot_fd_comparison_across_datasets(
            plot_data, seg_metric, fd_metric, show_rand_opt=True, alpha=0.8
        )
    elif isinstance(fd_metric, list):
        return plot_fd_comparison_across_datasets_multi(
            plot_data, seg_metric, fd_metric, show_rand_opt=True, alpha=0.8
        )
    else:
        raise ValueError


def make_figure_overall_comparison_ood(fd_results, ood_metric):
    plot_data = filter_expts_for_comparison(fd_results)
    fig = plot_ood_comparison_across_datasets(plot_data, ood_metric)
    return fig


def make_figure_aggregation_comparison(fd_results, expt_configs, seg_metric, fd_metric):
    expt_name_confid_combinations = [
        ("baseline-all_simple", "predictive_entropy_mean"),
        ("baseline-all_simple", "predictive_entropy_patch_min"),
        ("baseline-all_simple", "predictive_entropy_only_non_boundary"),
        ("baseline-predictive_entropy+heuristic", MAIN_SEG_METRIC),
        ("baseline-predictive_entropy+radiomics", MAIN_SEG_METRIC),
        ("deep_ensemble-all_simple", "predictive_entropy_mean"),
        ("deep_ensemble-all_simple", "predictive_entropy_patch_min"),
        ("deep_ensemble-all_simple", "predictive_entropy_only_non_boundary"),
        ("deep_ensemble-all_simple", "pairwise_dice"),  # NOTE legacy name
        ("deep_ensemble-predictive_entropy+heuristic", MAIN_SEG_METRIC),
        ("deep_ensemble-predictive_entropy+radiomics", MAIN_SEG_METRIC),
    ]
    if MAIN_SEG_METRIC == "generalized_dice":
        expt_name_confid_combinations += [("deep_ensemble-all_simple", "pairwise_gen_dice")]
    elif MAIN_SEG_METRIC == "mean_dice":
        expt_name_confid_combinations += [("deep_ensemble-all_simple", "pairwise_mean_dice")]
    # NOTE I missed patch_min in most of the experiments (except mcdropout in some cases).
    # Need to re-run these experiments
    plot_data = filter_expts_for_comparison(fd_results, expt_name_confid_combinations)
    fig = plot_fd_comparison_across_datasets(
        plot_data,
        seg_metric,
        fd_metric,
        change_markers=True,
        jitter=0,
        alpha=0.8,
        show_rand_opt=True,
    )
    # Some legend adjustments *rolleyes*
    ax = fig.get_axes()[0]
    handles = ax.get_legend().legend_handles
    labels = [txt._text for txt in ax.get_legend().texts]
    handles.insert(0, plt.Line2D([], [], marker="None", linestyle="None", color="None"))
    labels.insert(0, "")
    ax.legend(
        handles,
        labels,
        loc="upper left",
        ncols=2,
        bbox_to_anchor=(0, 1),
        title=None,
        frameon=True,
    )
    return fig


def plot_segmentation_performances(
    fd_results, expt_configs, metric="dice", included_domains=None, seed=0, fold=0
):
    # Plot 1: segmentation results for each dataset
    baseline_expts = fd_results[
        fd_results.expt_name.str.contains(r"baseline-all_simple$", regex=True)
    ].expt_id.unique()
    found_idx = None
    for idx in baseline_expts:
        # just one seed
        if "hparams" in expt_configs[idx].datamodule:
            curr_fold = expt_configs[idx].datamodule.hparams.fold
        else:
            curr_fold = expt_configs[idx].datamodule.fold
        if expt_configs[idx].seed == GLOBAL_SEEDS[seed] and curr_fold == fold:
            found_idx = idx
            break
    if found_idx is None:
        return plt.subplots()
    curr_results = load_raw_results(expt_configs[idx].paths.output_dir)
    fig, ax = plt.subplots()
    plot_class_seg_metrics_across_domains(
        curr_results, metric, ax=ax, included_domains=included_domains
    )
    return fig


def make_figure_seg_performance_overview(fd_results, expt_configs, seed=0, fold=0, figsize=None):
    # filter: only keep one method and seed, ie baseline
    baseline_expts = fd_results[
        fd_results.expt_name.str.contains(r"baseline-all_simple$", regex=True)
    ].expt_id.unique()
    expt_list = []
    dataset_count = defaultdict(int)
    for idx in baseline_expts:
        if "hparams" in expt_configs[idx].datamodule:
            dataset_id = expt_configs[idx].datamodule.hparams.dataset_id
            curr_fold = expt_configs[idx].datamodule.hparams.fold
        else:
            dataset_id = expt_configs[idx].datamodule.dataset_id
            curr_fold = expt_configs[idx].datamodule.fold
        if expt_configs[idx].seed == GLOBAL_SEEDS[seed] and curr_fold == fold:
            expt_list.append(idx)
            dataset_count[dataset_id] += 1
    if max(dataset_count.values()) > 1:
        print(dataset_count)
        raise ValueError("Found duplicate entries. Splitting by expt_id additionally")
    plot_data = []
    id_domains = {}
    for idx in expt_list:
        if "hparams" in expt_configs[idx].datamodule:
            dataset_id = expt_configs[idx].datamodule.hparams.dataset_id
        else:
            dataset_id = expt_configs[idx].datamodule.dataset_id
        dataset_id = int(dataset_id)
        curr_results = load_raw_results(expt_configs[idx].paths.output_dir)
        curr_results["expt_id"] = idx
        curr_results["dataset"] = dataset_id
        curr_id_domain = expt_configs[idx].dataset.get("id_domain", "ID")
        if dataset_id == 503:
            curr_id_domain = "HGG"  # though there also LGG in the training set
        if isinstance(curr_id_domain, str):
            curr_id_domain = [curr_id_domain]
        if curr_id_domain is None:
            curr_id_domain = []
        id_domains[dataset_id] = curr_id_domain
        plot_data.append(curr_results)
    plot_data = pd.concat(plot_data, ignore_index=True)
    # remove the low- and medium- domains from dataset 500
    plot_data = plot_data[
        ~(plot_data.domain.str.endswith("-low") | plot_data.domain.str.endswith("-medium"))
        | ~(plot_data.dataset == 500)
    ]
    if plot_data.groupby(["expt_name", "case_id"]).size().max() > 1:
        raise ValueError("Found duplicate entries. Splitting by expt_id additionally")
    plot_data = plot_data.sort_values("dataset")
    plot_data["id_label"] = plot_data.apply(
        lambda row: "ID" if row["domain"] in id_domains[row["dataset"]] else "Dataset shift",
        axis=1,
    )

    metric = f"metric_{MAIN_SEG_METRIC}"
    plot_data["dataset"] = plot_data.dataset.map(dataset_mapper)
    if figsize is None:
        figsize = get_figsize(aspect_ratio=0.5)
    fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(
        data=plot_data,
        x="dataset",
        y=metric,
        hue="id_label",
        dodge=True,
        ax=ax,
        order=[x for x in dataset_mapper.values() if x in plot_data.dataset.unique()],
        hue_order=["ID", "Dataset shift"],
        palette=["gray"] * 2,
        alpha=0.25,
        jitter=0.2,
        zorder=0,
    )
    sns.boxplot(
        data=plot_data,
        x="dataset",
        y=metric,
        hue="id_label",
        ax=ax,
        order=[x for x in dataset_mapper.values() if x in plot_data.dataset.unique()],
        hue_order=["ID", "Dataset shift"],
        fliersize=0,
        whis=(5, 95),
        fill=False,
        palette=["black", "indianred"],
    )
    # only show the legend entries for the box plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[2:], labels[2:], bbox_to_anchor=(0.5, 0.125), loc="upper left")
    # The areas look weird for kidney tumor...
    # sns.violinplot(
    #     data=plot_data,
    #     x="dataset",
    #     y=metric,
    #     hue="id_label",
    #     ax=ax,
    #     order=[x for x in dataset_mapper.values() if x in plot_data.dataset.unique()],
    #     hue_order=["ID", "Dataset shift"],
    #     cut=0,
    #     inner=None,
    #     fill=False,
    #     density_norm="area",
    # )
    # remove the legend title
    ax.get_legend().set_title("")
    ax.set_ylabel(MAIN_SEG_METRIC)
    return fig


def make_figure_seg_model_comparison(fd_results, expt_configs, seed=0):
    metric, plot_data = prepare_segmodel_comparison(fd_results, expt_configs, seed)
    # average across folds
    plot_data = plot_data.groupby(["model", "dataset", "case_id"])[metric].mean().reset_index()
    if plot_data.groupby(["model", "case_id"]).size().max() > 1:
        raise ValueError("Found duplicate entries. Splitting by expt_id additionally")
    hue_order = ["Single Network", "MC-Dropout", "Deep Ensemble"]
    hue_colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=get_figsize(aspect_ratio=0.5))
    sns.stripplot(
        data=plot_data,
        x="dataset",
        y=metric,
        hue="model",
        dodge=True,
        ax=ax,
        order=[x for x in dataset_mapper.values() if x in plot_data.dataset.unique()],
        hue_order=hue_order,
        palette=["gray"] * len(hue_order),
        alpha=0.25,
        jitter=0.2,
        zorder=0,
    )
    sns.boxplot(
        data=plot_data,
        x="dataset",
        y=metric,
        hue="model",
        ax=ax,
        order=[x for x in dataset_mapper.values() if x in plot_data.dataset.unique()],
        hue_order=hue_order,
        palette=hue_colors,
        fliersize=0,
        whis=(5, 95),
        fill=False,
        gap=0.1,
    )
    # only show the legend entries for the box plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[len(hue_order) :],
        labels[len(hue_order) :],
        bbox_to_anchor=(0.5, 0.125),
        loc="upper left",
    )
    # The areas look weird for kidney tumor...
    # sns.violinplot(
    #     data=plot_data,
    #     x="dataset",
    #     y=metric,
    #     hue="id_label",
    #     ax=ax,
    #     order=[x for x in dataset_mapper.values() if x in plot_data.dataset.unique()],
    #     hue_order=["ID", "Dataset shift"],
    #     cut=0,
    #     inner=None,
    #     fill=False,
    #     density_norm="area",
    # )
    # remove the legend title
    ax.get_legend().set_title("")
    ax.set_ylabel(MAIN_SEG_METRIC)
    return fig


def make_table_seg_model_comparison(fd_results, expt_configs, output_file: Path, seed=0):
    metric, plot_data = prepare_segmodel_comparison(fd_results, expt_configs, seed)
    # compute mean and sd across folds
    assert np.all(plot_data.groupby(["model", "dataset", "case_id"]).size() == 5)
    plot_data = (
        plot_data.groupby(["model", "dataset"])[metric].describe()[["mean", "std"]].reset_index()
    )
    plot_data = plot_data.pivot(index="model", columns="dataset")
    # swap the column levels
    plot_data = plot_data.swaplevel(axis=1).sort_index(axis=1)
    plot_data = plot_data.loc[
        ["Single Network", "MC-Dropout", "Deep Ensemble"],
        ["Brain tumor (2D)", "Brain tumor", "Heart", "Kidney tumor", "Covid", "Prostate"],
    ]
    idx = pd.IndexSlice
    # mean_slice_ = idx[:, idx[:, "mean"]]
    std_slice_ = idx[:, idx[:, "std"]]
    plot_data *= 100
    s = plot_data.style.format(precision=1, na_rep="n/a")
    s.set_properties(subset=std_slice_, **{"color": "gray"})
    # either highlight min or color map
    # cm = sns.color_palette("YlGn", as_cmap=True)
    # s.background_gradient(cmap=cm, axis=0, subset=mean_slice_)
    # s.hide(axis=1, level=1)
    s.to_html(
        output_file.with_suffix(".html"),
        hrules=True,
        multicol_align="c",
    )
    s.to_latex(
        output_file,
        caption="INSERT CAPTION.",
        column_format="l" + "rl" * 6,
        clines="skip-last;data",
        convert_css=True,
        position_float="centering",
        multicol_align="c",
        hrules=True,
    )
    return s


def prepare_segmodel_comparison(fd_results, expt_configs, seed):
    metric = f"metric_{MAIN_SEG_METRIC}"
    expt_list = []
    for seg_model in ["baseline-all_simple", "mcdropout-all_simple", "deep_ensemble-all_simple"]:
        baseline_expts = fd_results[
            fd_results.expt_name.str.contains(f"{seg_model}$", regex=True)
        ].expt_id.unique()
        dataset_count = defaultdict(int)
        for idx in baseline_expts:
            if "hparams" in expt_configs[idx].datamodule:
                dataset_id = expt_configs[idx].datamodule.hparams.dataset_id
            else:
                dataset_id = expt_configs[idx].datamodule.dataset_id
            if expt_configs[idx].seed == GLOBAL_SEEDS[seed]:
                expt_list.append(idx)
                dataset_count[dataset_id] += 1
        if not all(x == 5 for x in dataset_count.values()):
            print(dataset_count)
            raise ValueError("Expected 5 folds!")
    plot_data = []
    for idx in expt_list:
        if "hparams" in expt_configs[idx].datamodule:
            dataset_id = expt_configs[idx].datamodule.hparams.dataset_id
        else:
            dataset_id = expt_configs[idx].datamodule.dataset_id
        curr_results = load_raw_results(expt_configs[idx].paths.output_dir)
        curr_results["expt_id"] = idx
        curr_results["dataset"] = int(dataset_id)
        plot_data.append(curr_results)
    plot_data = pd.concat(plot_data, ignore_index=True)
    plot_data = plot_data[["case_id", "expt_name", "dataset", "domain", metric]]
    # remove the low- and medium- domains from dataset 500
    plot_data = plot_data[
        ~(plot_data.domain.str.endswith("-low") | plot_data.domain.str.endswith("-medium"))
        | ~(plot_data.dataset == 500)
    ]
    plot_data = plot_data.sort_values("dataset")
    plot_data["dataset"] = plot_data.dataset.map(dataset_mapper)

    def quick_mapper(x: str):
        if x.endswith("baseline-all_simple"):
            return "Single Network"
        elif x.endswith("mcdropout-all_simple"):
            return "MC-Dropout"
        elif x.endswith("deep_ensemble-all_simple"):
            return "Deep Ensemble"
        else:
            return x

    plot_data["model"] = plot_data.expt_name.map(quick_mapper)
    return metric, plot_data


def plot_evaluation_fig1(output_dir: Path):
    # Artificial risk vs confidence
    num_total = 100
    num_outliers = 30
    np.random.seed(4242)
    confids = np.random.rand(num_total)
    risks = 1 - confids + np.random.randn(num_total) * 0.1
    risks[np.random.randint(0, num_total, num_outliers)] = np.random.rand(num_outliers)
    risks = np.clip(risks, 0, 1)

    fig = _fig1_risk_vs_confid(confids, risks)
    save_and_close(fig, output_dir / "risk_confid_fig1.png")

    fig = _fig1_risk_vs_coverage(confids, risks)
    save_and_close(fig, output_dir / "risk_coverage_fig1.png")


def _fig1_risk_vs_coverage(confids, risks):
    from segmentation_failures.evaluation.failure_detection.metrics import (
        StatsCache,
        get_metric_function,
    )

    color_accept = "#147012"
    optimal_confid = -risks
    stats_cache = StatsCache(confids, risks)
    opt_stats = StatsCache(optimal_confid, risks)

    opt_coverages, opt_selrisks, _ = opt_stats.rc_curve_stats
    coverages, selrisks, _ = stats_cache.rc_curve_stats
    aurc = get_metric_function("aurc")(stats_cache)
    # plot the risks
    fig = plt.figure(figsize=(3, 3))
    # set aspect ratio to 1:1
    fig.patch.set_facecolor("none")
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(coverages, selrisks, color=color_accept)
    # add a shaded region below the curve
    plt.fill_between(
        coverages, 0, selrisks, color=color_accept, alpha=0.2, label=f"AURC = {aurc:0.3f}"
    )
    plt.plot(opt_coverages, opt_selrisks, linestyle=":", color="black", label="optimal")
    plt.xlabel(r"Coverage (% samples > $\tau$)")
    plt.ylabel(r"Avg. risk (of samples > $\tau$)")
    plt.legend()
    return fig


def _fig1_risk_vs_confid(confids, risks):
    color_accept = "#147012"
    color_reject = "#cc0000"
    fig, ax = plt.subplots(figsize=(3, 3), layout="constrained")
    # make sure the aspect ratio is square
    ax.set_aspect("equal")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Risk")
    confid_thresh = 0.6
    # plt.plot(confids, risks, ".", color="gray")
    # overlay shaded areas of different colors left and right of the confidence threshold
    plt.axvspan(-1, confid_thresh, alpha=0.2, color=color_reject)
    plt.axvspan(confid_thresh, 2, alpha=0.2, color=color_accept)
    # plot the points in the same color as the shaded areas
    plt.scatter(
        confids[confids < confid_thresh], risks[confids < confid_thresh], color=color_reject
    )
    plt.scatter(
        confids[confids >= confid_thresh], risks[confids >= confid_thresh], color=color_accept
    )
    plt.axvline(x=confid_thresh, color="black", linestyle="--")
    # replace the xtick at the confidence threshold with a text label (tau)
    plt.xticks([0, confid_thresh, 1], [0, "$\\tau$", 1])
    # save with transparent background
    fig.patch.set_facecolor("none")
    return fig


def make_progress_table(expt_configs, output_dir):
    # for each dataset, make a table where each row is an experiment name and the columns are the seeds
    dataset_tables = defaultdict(lambda: pd.DataFrame())
    seed_inv_mapping = {v: k for k, v in GLOBAL_SEEDS.items()}
    for cfg in expt_configs.values():
        dataset_id = (
            int(cfg.datamodule.hparams.dataset_id)
            if "hparams" in cfg.datamodule
            else cfg.datamodule.dataset_id
        )
        curr_table = dataset_tables[dataset_id]
        # nicer expt name
        expt_name = cfg.expt_name
        if "unet_dropout" in expt_name:
            # remove everything before unet_dropout- (including)
            expt_name = expt_name.split("unet_dropout-")[-1]
            # expt_name = map_method_names(expt_name)
        seed = f"seed {seed_inv_mapping[cfg.seed]}"
        # if there is an entry already, add 1 to the seed column
        if expt_name in curr_table.index and seed in curr_table.columns:
            if pd.isna(curr_table.loc[expt_name, seed]):
                curr_table.loc[expt_name, seed] = 1
            else:
                curr_table.loc[expt_name, seed] += 1
        else:
            curr_table.loc[expt_name, seed] = 1
    for ds_id, df in dataset_tables.items():
        df = df.sort_index(axis=0)
        df = df.sort_index(axis=1)
        df.fillna("0", inplace=True)
        df = df.astype(int)
        df = df.sort_index()
        # df.to_html(output_dir / f"progress_table_{ds_id}.html")
        html = df.to_html()
        html_with_title = f"<h2>Dataset {ds_id}</h2>" + html
        # Write the HTML string to a file
        with open(output_dir / f"progress_table_{ds_id}.html", "w") as f:
            f.write(html_with_title)


def export_experiment_list_qual_analysis(fd_results, output_dir):
    # for every dataset, export a csv with the experiment names and the corresponding expt_id
    # for selected experiments
    pw_dice_name = (
        "pairwise_gen_dice" if MAIN_SEG_METRIC == "generalized_dice" else "pairwise_mean_dice"
    )
    selected_expts = [
        ("deep_ensemble-all_simple", "pairwise_dice"),  # legacy experiments
        ("deep_ensemble-all_simple", "dice_0"),  # binary seg datasets
        ("deep_ensemble-all_simple", pw_dice_name),
        ("deep_ensemble-quality_regression", MAIN_SEG_METRIC),
    ]
    df_subset = filter_expts_for_comparison(fd_results, selected_expts)
    for (dataset_id, fold), dataset_results_df in df_subset.groupby(["dataset", "fold"]):
        expt_list = dataset_results_df.root_dir.unique().tolist()
        with open(output_dir / f"expt_list_dataset{dataset_id}_fold{fold}.json", "w") as f:
            json.dump(expt_list, f)


def compute_bootstrapped_results(
    fd_results,
    seg_metric=MAIN_SEG_METRIC,
    fd_metrics=None,
    n_bootstrap=500,
    allow_duplicates=False,
):
    if fd_metrics is None:
        fd_metrics = ["aurc", "e-aurc"]
    assert fd_results.dataset.nunique() == 1
    dataset_id = fd_results.dataset.unique()[0]
    # define which experiments (name + confidence name) we want to rank
    filtered_data = filter_expts_for_comparison(fd_results)
    filtered_data = filtered_data[filtered_data.metric == seg_metric]
    # make the experiment name nicer
    filtered_data["nice_expt_name"] = get_unique_expt_name(
        filtered_data.expt_name, filtered_data.confid_name
    )
    if seg_metric.startswith("mean_"):
        metric_info = get_metrics_info()[seg_metric.removeprefix("mean_")]
    else:
        metric_info = get_metrics_info()[seg_metric]
    # Perform bootstrapping
    # NOTE do this only once, to get a list of test sets. These should be used for all experiments!
    if "n_cases" in filtered_data.columns and not filtered_data.n_cases.isna().any():
        assert filtered_data.n_cases.nunique() == 1
        n_cases = int(filtered_data.n_cases.iloc[0])
    else:
        n_cases = get_num_test_cases(dataset_id)
    bootstrap_samples = np.random.choice(
        np.arange(n_cases), size=(n_bootstrap, n_cases), replace=True
    )  # these are indices
    # For each test set, perform the evaluation and ranking
    results = []
    for expt_name, group_df in filtered_data.groupby("nice_expt_name"):
        if len(group_df) > 1:
            if not allow_duplicates:
                raise ValueError(
                    f"Found several experiments for one name: {group_df.root_dir.tolist()}"
                )
            logger.warning(
                f"Found {len(group_df)} results for {expt_name}. "
                "Maybe some seeds were re-run? Consider dropping them manually."
            )
            # pick the latest version
            group_df = group_df.sort_values("expt_version", ascending=False).iloc[0:1]
        expt_dir = Path(group_df.root_dir.iloc[0])
        confid_name = group_df.confid_name.iloc[0]
        # load the raw data for each experiment (ie risks and confidence scores for each case)
        expt_data = ExperimentData.from_experiment_dir(expt_dir)
        for boot_idx, test_set in enumerate(bootstrap_samples):
            # ie method
            confid_scores = expt_data.confid_scores[
                test_set, expt_data.confid_scores_names.index(confid_name)
            ]
            seg_metrics = expt_data.segmentation_metrics[
                test_set, expt_data.segmentation_metrics_names.index(seg_metric)
            ]
            fd_scores, _ = compute_fd_scores(
                confid_scores,
                seg_metrics,
                metric_info,
                fd_metrics,
                failure_thresh=0.5,  # not used if not failauc in metrics
            )
            result_dict = {
                "bootstrap": boot_idx,
                "method": expt_name,
            }
            result_dict.update(fd_scores)
            results.append(result_dict)
    # format: dataframe with columns bootstrap_idx, method_name, *fd_metrics
    return pd.DataFrame(results)


def main(expt_group_dir: str, output_dir: str):
    expt_group_dir = Path(expt_group_dir)
    output_dir = Path(output_dir)
    included_datasets = [500, 503, 511, 515, 520, 521]
    logger.remove()  # Remove default 'stderr' handler
    logger.add(sys.stderr, level="INFO")
    # Load results for all datasets
    output_dir.mkdir(exist_ok=True)
    # caching is mainly for debugging
    tmp_fd_results_cache = output_dir / "tmp_fd_results_cache.csv"
    tmp_config_cache = output_dir / "tmp_ood_results_cache.csv"
    tmp_ood_results_cache = output_dir / "tmp_config_cache.pkl"
    if tmp_fd_results_cache.exists():
        # NOTE delete the cache files if you want to recompute the results
        fd_results = pd.read_csv(tmp_fd_results_cache)
        ood_results = pd.read_csv(tmp_ood_results_cache)
        with open(tmp_config_cache, "rb") as f:
            expt_configs = pickle.load(f)
    else:
        fd_results, ood_results, expt_configs = load_results(
            expt_group_dir, included_datasets=included_datasets
        )
        fd_results.to_csv(tmp_fd_results_cache, index=False)
        ood_results.to_csv(tmp_ood_results_cache, index=False)
        with open(tmp_config_cache, "wb") as f:
            pickle.dump(expt_configs, f)

    # invert the nAURC metric -> still in (0, 1) but 0 is best
    fd_results["norm-aurc"] = 1 - fd_results["norm-aurc"]

    # These steps should not have an effect in the latest results
    # filter previous runs
    fd_results = fd_results[
        fd_results.expt_name.str.contains("mahalanobis|vae_", regex=True)
        | (fd_results.expt_version.astype(int) >= 23960389)
    ]
    # there could be duplicate experiments (ie different experiment versions)
    fd_results = fd_results.drop_duplicates(
        subset=[
            x
            for x in fd_results.columns
            if x not in ["expt_version", "expt_id", "root_dir", "file_risk_coverage_curve"]
        ]
    )
    # apply changes to expt configs
    expt_configs = {k: v for k, v in expt_configs.items() if k in fd_results.expt_id.unique()}
    fd_results_all_domain = fd_results[fd_results.domain == "all_"]
    if any(x.endswith("-low") or x.endswith("-medium") for x in fd_results.domain.unique()):
        raise ValueError("Found low or medium domains in the results.")

    # --- Figures ---
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.7)
    plt.rcParams["font.family"] = "Liberation Sans"

    # For checking how many test results I have for each dataset/method use:
    # check_training_progress.ipynb

    with sns.axes_style("ticks"):
        plot_evaluation_fig1(output_dir=output_dir)

    # FIG segmentation performance for all datasets
    curr_output_dir = output_dir / "seg_performance"
    curr_output_dir.mkdir(exist_ok=True)
    for seg_metric in ["dice", "surface_dice"]:
        for dataset_id, df in fd_results.groupby("dataset"):
            included_domains = None
            if dataset_id == 500:
                included_domains = [
                    x
                    for x in df.domain.unique()
                    if not (
                        x.endswith("-low") or x.endswith("-medium") or x in ["all_", "all_ood_"]
                    )
                ]
            fig = plot_segmentation_performances(
                df, expt_configs, metric=seg_metric, included_domains=included_domains
            )
            save_and_close(fig, curr_output_dir / f"{dataset_id}_{seg_metric}_seed0_fold0.png")
    fig = make_figure_seg_performance_overview(fd_results, expt_configs)
    save_and_close(fig, output_dir / f"seg_performance_id_ood_{MAIN_SEG_METRIC}.png")
    fig = make_figure_seg_model_comparison(fd_results, expt_configs)
    save_and_close(fig, output_dir / f"seg_performance_model_comparison_{MAIN_SEG_METRIC}.png")

    # FIG Bootstrapping: (actual plotting done in bootstrap_ranking.py)
    for fold in range(5):
        bootstrap_dir = output_dir / f"bootstrapping_fold{fold}"
        bootstrap_dir.mkdir(exist_ok=True)
        for dataset_id in fd_results_all_domain.dataset.unique():
            subdf = fd_results_all_domain[
                (fd_results_all_domain.dataset == dataset_id)
                & (fd_results_all_domain.fold == fold)
            ]
            assert subdf.seed.nunique() == 1
            for seg_metric in [MAIN_SEG_METRIC, "mean_surface_dice"]:
                curr_out_dir = bootstrap_dir / seg_metric
                curr_out_dir.mkdir(exist_ok=True)
                results = compute_bootstrapped_results(
                    subdf,
                    seg_metric=seg_metric,
                    fd_metrics=["aurc", "spearman"],
                    allow_duplicates=False,
                    n_bootstrap=500,
                )
                results.to_csv(curr_out_dir / f"dataset{dataset_id:03d}.csv", index=False)

    # FIG: AURC (or other FD metrics) for different datasets and methods
    curr_output_dir = output_dir / "overview_all"
    curr_output_dir.mkdir(exist_ok=True)
    fd_metric = "aurc"
    fig = make_figure_overall_comparison(
        fd_results_all_domain,
        expt_configs,
        seg_metric=MAIN_SEG_METRIC,
        fd_metric=fd_metric,
        include_ranking=True,
    )
    save_and_close(
        fig, curr_output_dir / f"overview_{fd_metric}_{MAIN_SEG_METRIC}_all_with_ranking.png"
    )
    fig = make_figure_overall_comparison(
        fd_results_all_domain,
        expt_configs,
        seg_metric=MAIN_SEG_METRIC,
        fd_metric=["norm-aurc", "spearman", "pearson"],
    )
    save_and_close(fig, curr_output_dir / f"overview_multi-fd-metric_{MAIN_SEG_METRIC}_all.png")

    # FIG AURC (or other FD metrics) for different datasets and aggregation methods
    curr_output_dir = output_dir / "overview_agg"
    curr_output_dir.mkdir(exist_ok=True)
    for fd_metric in ["aurc", "norm-aurc", "spearman", "pearson"]:
        fig = make_figure_aggregation_comparison(
            fd_results_all_domain,
            expt_configs,
            seg_metric=MAIN_SEG_METRIC,
            fd_metric=fd_metric,
        )
        save_and_close(fig, curr_output_dir / f"overview_{fd_metric}_{MAIN_SEG_METRIC}_agg.png")

    # Risk coverage curves for all datasets
    curve_dir = output_dir / "risk_coverage_curves"
    curve_dir.mkdir(exist_ok=True)
    for dataset in included_datasets:
        dataset_results = fd_results_all_domain[fd_results_all_domain.dataset == dataset]
        if dataset_results.empty:
            continue
        fig = plot_risk_coverage_curves(dataset_results, risk_metric=MAIN_SEG_METRIC)
        save_and_close(fig, curve_dir / f"dataset{dataset:03d}.png")

    # OOD-AUC similar to AURC plot
    fig = make_figure_overall_comparison_ood(ood_results, ood_metric="ood_auc")
    save_and_close(fig, output_dir / "overview_all" / "overview_ood-auc.png")

    # Compare Dice vs confidence for selected experiments
    output_subdir = output_dir / "dice_vs_confid"
    output_subdir.mkdir(exist_ok=True)
    for _, df in fd_results.groupby("dataset"):
        make_plots_confid_vs_metric(
            df, expt_configs, metric=MAIN_SEG_METRIC, output_dir=output_subdir
        )

    # --- Tables ---
    # TABLE: Compare mean Dice to Surface Dice
    make_table_dice_vs_surf_dist(
        fd_results_all_domain, "aurc", output_dir / "table_dice_vs_surface_dist.html"
    )

    # TABLE: Compare AURC of all methods
    make_table_big_comparison(
        fd_results_all_domain, "aurc", MAIN_SEG_METRIC, output_dir / "table_big_comparison.tex"
    )
    make_table_big_comparison_multiple_metrics(
        fd_results_all_domain,
        ["aurc", "spearman", "pearson"],
        MAIN_SEG_METRIC,
        output_dir / "table_big_comparison_multimetric.tex",
    )

    # TABLE: Compare mean DSC of single network, mcdropout, deep ensemble
    make_table_seg_model_comparison(
        fd_results_all_domain, expt_configs, output_dir / "table_seg_model_comparison.tex"
    )

    # --- Independent qualitative analysis ---
    # export csvs of experiment directories for manual inspection
    output_dir_qual_analysis = output_dir / "qualitative_analysis"
    output_dir_qual_analysis.mkdir(exist_ok=True)
    export_experiment_list_qual_analysis(fd_results, output_dir_qual_analysis)

    return fd_results, ood_results, expt_configs


if __name__ == "__main__":
    expt_group_dir = "/mnt/E132-Projekte/Projects/2023_MaxZ_segmentation_failures/cluster_logs/logs/paper_expts_2403"
    output_dir = "/home/m167k/Projects/segmentation_failures/gitlab_segfail/analysis/figures/testing_publication"
    main(expt_group_dir, output_dir)
