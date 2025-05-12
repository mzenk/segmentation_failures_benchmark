"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import re
import shutil
from itertools import product
from pathlib import Path
from typing import Dict

import SimpleITK as sitk
import torch
import torchio as tio
from loguru import logger
from tqdm import tqdm

from segmentation_failures.data.corruptions.image_corruptions_tio import (
    TransformRegistry,
)
from segmentation_failures.utils.io import load_json, save_json

TRANSFORMED_PREFIX = "TRF_"


# TODO more checks?
def check_input_data(image_dir: Path, label_dir: Path):
    assert image_dir.exists(), f"Image directory {image_dir} does not exist."
    assert label_dir.exists(), f"Label directory {label_dir} does not exist."


def clean_output_dirs(image_dir: Path, label_dir):
    # Delete all previously transformed images and labels
    for img_path in image_dir.glob("*.nii.gz"):
        if img_path.name.startswith(TRANSFORMED_PREFIX):
            img_path.unlink()
    for label_path in label_dir.glob("*.nii.gz"):
        if label_path.name.startswith(TRANSFORMED_PREFIX):
            label_path.unlink()


def process_subject(
    subject: tio.Subject,
    transform: tio.Transform,
    img_target_dir: Path,
    seg_target_dir: Path,
    file_prefix: str,
):
    trf_subject = transform(subject)
    for k, img in trf_subject.get_images_dict(intensity_only=False).items():
        out_filename = file_prefix + img.path.name
        out_path = img_target_dir / out_filename
        if k == "seg":
            out_path = seg_target_dir / out_filename
        sitk.WriteImage(img.as_sitk(), str(out_path.absolute()))


def setup_dataset(
    image_dir: Path,
    label_dir: Path,
    modalities: Dict[str, str],
) -> tio.SubjectsDataset:
    subjects = []
    for label_path in label_dir.glob("*.nii.gz"):
        if label_path.name.startswith(TRANSFORMED_PREFIX):
            # this prefix indicates the image has already been corrupted. Tt should not be included in the data to corrupt again
            continue
        case_id = re.findall(r"(\w+)\.nii\.gz", label_path.name)[0]
        subj_dict = {"case_id": case_id, "seg": tio.LabelMap(label_path)}
        # NOTE just for debugging
        for mod_idx, mod_name in modalities.items():
            img_path = image_dir / f"{case_id}_{int(mod_idx):04d}.nii.gz"
            subj_dict[mod_name] = tio.ScalarImage(img_path)
        subjects.append(tio.Subject(subj_dict))
    return tio.SubjectsDataset(subjects)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--data_dir",
        help="Path where data that should be corrupted is currently saved. Assumes default dataset structure.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Path where transformed data will be saved (in subdirectories). "
        "If not specified, will be saved in the same folder as the original data.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-t",
        "--transform",
        help="Which transform(s) to apply. Multiple can be specified (separated by space)",
        choices=TransformRegistry.list_transforms(),
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-m",
        "--magnitude",
        type=str,
        help=(
            "Which transform magnitudes(s) to use. Multiple can be specified (separated by space)."
            "The specified magnitudes will be done for each transform."
        ),
        choices=["low", "medium", "high"],
        default=["high"],
        nargs="+",
    )
    parser.add_argument(
        "--seed",
        help="Random seed.",
        type=int,
        default=49871346,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    case2domain_map = {}
    json_output = output_dir / "domain_mapping_00.json"
    if json_output.exists():
        case2domain_map.update(load_json(json_output))

    trf_params, add_case2domain_map = corrupt_data(
        data_path, output_dir, args.transform, args.magnitude
    )
    case2domain_map.update(add_case2domain_map)
    save_json(case2domain_map, json_output)
    save_json(trf_params, output_dir / "transform_params.json")


def corrupt_data(data_path, output_dir, transform_list, magnitude_list, overwrite_ok=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite_ok:
            raise FileExistsError(f"The output directory {output_dir} already exists. Aborting.")
        logger.warning("The output directory is not empty! Overwrites are possible.")
    output_dir.mkdir(parents=True, exist_ok=True)

    images_ts_dir = data_path / "imagesTs"
    labels_ts_dir = data_path / "labelsTs"
    check_input_data(images_ts_dir, labels_ts_dir)
    output_img_dir = output_dir / "imagesTs"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_seg_dir = output_dir / "labelsTs"
    output_seg_dir.mkdir(parents=True, exist_ok=True)
    if images_ts_dir != output_img_dir:
        # copy original images to output directory
        for img_path in images_ts_dir.glob("*.nii.gz"):
            shutil.copy(img_path, output_img_dir)
    if labels_ts_dir != output_seg_dir:
        # copy original labels to output directory
        for label_path in labels_ts_dir.glob("*.nii.gz"):
            shutil.copy(label_path, output_seg_dir)
    clean_output_dirs(output_img_dir, output_seg_dir)

    # setup dataset
    modalities = load_json(data_path / "dataset.json")["channel_names"]
    dataset = setup_dataset(images_ts_dir, labels_ts_dir, modalities)

    case2domain_map = {}
    trf_params = {}
    for subj in tqdm(dataset):
        case2domain_map[subj.case_id] = "noshift"  # in-distribution cases copied earlier
        for trf_name, trf_magn in product(transform_list, magnitude_list):
            domain = f"{trf_name}-{trf_magn}"
            case_modifier = f"{TRANSFORMED_PREFIX}{domain}_"
            new_case_id = f"{case_modifier}{subj.case_id}"
            case2domain_map[new_case_id] = domain
            trf, curr_params = TransformRegistry.get_transform(trf_name, trf_magn)
            trf_params[domain] = curr_params
            process_subject(subj, trf, output_img_dir, output_seg_dir, case_modifier)
    return trf_params, case2domain_map


if __name__ == "__main__":
    main()
