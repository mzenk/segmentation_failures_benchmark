"""
What I want to do here:
- select slice that contains the center of mass of the image
- downsample to smaller resolution (optional)
- split into train/test? Or use validation split(s)?
- convert to nnunet-like file tree structure (imagesTr, labelsTr, ...)

"""

import os
from argparse import ArgumentParser
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import skimage.transform as trf
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from segmentation_failures.data.corruptions.corrupt_data_torchio import corrupt_data
from segmentation_failures.utils.data import kfold_split
from segmentation_failures.utils.io import load_json, save_json

TASK_NAME = "Dataset500_simple_fets_corruptions"


# =====
# adapted from nnunet
def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes

    assert (
        len(data.shape) == 4 or len(data.shape) == 3
    ), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (
        slice(bbox[0][0], bbox[0][1]),
        slice(bbox[1][0], bbox[1][1]),
        slice(bbox[2][0], bbox[2][1]),
    )
    return image[resizer]


def crop_to_nonzero(data: np.ndarray, seg: np.ndarray = None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    # # not sure why this should be done in my case
    # nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    # if seg is not None:
    #     seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    # else:
    #     nonzero_mask = nonzero_mask.astype(int)
    #     nonzero_mask[nonzero_mask == 0] = nonzero_label
    #     nonzero_mask[nonzero_mask > 0] = 0
    #     seg = nonzero_mask
    return data, seg, bbox


# =====


def process_case_crop_and_pad(
    case_dir,
    target_size: int,
    output_dir_img,
    output_dir_lab,
    meta_dict,
):
    case_id = case_dir.name
    images = {
        "t1": case_dir / f"{case_id}_t1.nii.gz",
        "t1ce": case_dir / f"{case_id}_t1ce.nii.gz",
        "t2": case_dir / f"{case_id}_t2.nii.gz",
        "flair": case_dir / f"{case_id}_flair.nii.gz",
    }
    modalities = list(meta_dict["channel_names"].values())
    # can filter if necessary
    images = {k: v for k, v in images.items() if k in modalities}
    seg_path = case_dir / f"{case_id}_seg.nii.gz"

    # read images, select slice and store slice index
    slice_idx = None
    seg_npy = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path))).astype(
        np.uint8
    )  # shape (z, y, x)
    assert (
        seg_npy.shape[-1] == seg_npy.shape[-2]
    ), "this script expects same x and y image dimensions"

    scaling_factor = target_size / seg_npy.shape[-1]
    multiseq_image = []
    for _, img_path in images.items():
        img_npy = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        multiseq_image.append(img_npy)
    multiseq_image = np.stack(multiseq_image)  # shape (c, z, y, x)

    # select slice with largest tumor extent
    slice_idx = np.argmax(np.sum(seg_npy > 0, axis=(1, 2)))  # TODO maybe add more than one slice
    seg_npy = seg_npy[[slice_idx]]
    multiseq_image = multiseq_image[:, [slice_idx]]

    if len(meta_dict["labels"]) == 2:
        # only whole tumor
        # TODO maybe only include T2-FLAIR in this case?
        seg_npy[seg_npy > 0] = 1
    else:
        # convert to nnunet style consecutive labels
        mapping = {
            0: 0,  # background
            1: 2,  # necrosis
            2: 1,  # edema
            4: 3,  # enhancing
        }
        seg_npy = np.vectorize(mapping.get)(seg_npy)

    cropped_image, cropped_seg, _ = crop_to_nonzero(multiseq_image, seg_npy[None])
    output_shape = (
        np.ceil(scaling_factor * cropped_image.shape[2]),
        np.ceil(scaling_factor * cropped_image.shape[3]),
    )
    # TODO generalize this (I use target_size = 80 so far):
    # Determine the maximum brain extent on all cropped slices (or maybe complete volumes)
    # and then scale the image by target_size / MAX_EXTENT. Afterwards it should be possible to pad all sides to target size.
    if output_shape[0] > 64 or output_shape[1] > 64:
        raise ValueError("Can only pad with positive values. Increase target size, please!")
    pad_width = [int(64 - s_out) for s_out in output_shape]
    pad_width = (
        (pad_width[0] // 2, pad_width[0] // 2 + (pad_width[0] % 2)),
        (pad_width[1] // 2, pad_width[1] // 2 + (pad_width[1] % 2)),
    )

    # -- image --
    for mod_idx in range(len(cropped_image)):
        resized_slice = trf.resize(
            cropped_image[mod_idx].squeeze(),
            output_shape=output_shape,
            order=1,
            anti_aliasing=True,
        )
        # padding
        resized_slice = np.pad(
            resized_slice, pad_width=pad_width, mode="constant", constant_values=0
        )
        sitk_image_new = sitk.GetImageFromArray(resized_slice)  # size (x, y)
        sitk.WriteImage(
            sitk_image_new,
            str(Path(output_dir_img) / f"{case_id}_{mod_idx:04d}.nii.gz"),
        )

    # -- Segmentation --
    resized_slice = trf.resize(
        cropped_seg.squeeze(), output_shape=output_shape, order=0, anti_aliasing=False
    )
    # padding segmentation
    resized_slice = np.pad(resized_slice, pad_width=pad_width, mode="constant", constant_values=0)
    sitk_image_new = sitk.GetImageFromArray(resized_slice)  # size (x, y)
    sitk.WriteImage(sitk_image_new, str(Path(output_dir_lab) / f"{case_id}.nii.gz"))

    return case_id, int(slice_idx)


def process_case(case_dir, target_shape, output_dir_img, output_dir_lab, meta_dict):
    case_id = case_dir.name
    images = {
        "t1": case_dir / f"{case_id}_t1.nii.gz",
        "t1ce": case_dir / f"{case_id}_t1ce.nii.gz",
        "t2": case_dir / f"{case_id}_t2.nii.gz",
        "flair": case_dir / f"{case_id}_flair.nii.gz",
    }
    seg_path = case_dir / f"{case_id}_seg.nii.gz"
    mod2code = {v: int(k) for k, v in meta_dict["modalities"].items()}

    # read flair, select slice and store slice index
    if isinstance(target_shape, int):
        target_shape = (target_shape, target_shape)
    slice_idx = None
    seg_npy = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path))).astype(
        np.uint8
    )  # shape (z, y, x)
    # select slice with largest tumor extent
    slice_idx = np.argmax(np.sum(seg_npy > 0, axis=(1, 2)))

    resized_slice = trf.resize(
        seg_npy[slice_idx], output_shape=target_shape, order=0, anti_aliasing=False
    )
    sitk_image_new = sitk.GetImageFromArray(resized_slice)  # size (x, y)
    sitk.WriteImage(sitk_image_new, str(Path(output_dir_lab) / f"{case_id}.nii.gz"))

    for mod, img_path in images.items():
        img_npy = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))  # shape (z, y, x)
        resized_slice = trf.resize(
            img_npy[slice_idx], output_shape=target_shape, order=1, anti_aliasing=True
        )
        sitk_image_new = sitk.GetImageFromArray(resized_slice)  # size (x, y)
        sitk.WriteImage(
            sitk_image_new,
            str(Path(output_dir_img) / f"{case_id}_{mod2code[mod]:04d}.nii.gz"),
        )
    return case_id, int(slice_idx)


def run_preprocessing_for_case_list(
    img_size, meta, images_tr_dir, labels_tr_dir, train_cases, num_processes=0
):
    case_slice_dict = {}
    if num_processes == 0:
        for case_dir in tqdm(train_cases):
            case_id, slice_idx = process_case_crop_and_pad(
                case_dir,
                img_size,
                images_tr_dir,
                labels_tr_dir,
                meta,
            )
            case_slice_dict[case_id] = slice_idx
    else:
        with Pool(processes=num_processes) as p:
            results = p.starmap(
                process_case_crop_and_pad,
                zip(
                    train_cases,
                    repeat(img_size),
                    repeat(images_tr_dir),
                    repeat(labels_tr_dir),
                    repeat(meta),
                ),
            )
        case_slice_dict.update({x[0]: x[1] for x in results})
    return case_slice_dict


def add_arguments(parser: ArgumentParser):
    parser.add_argument("--raw_data_path", type=str)
    parser.add_argument(
        "--out_path",
        type=str,
        default=os.environ["nnUNet_raw"] + "/" + TASK_NAME,
    )
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--img_size", type=int, default=80)  # ugly: results in 64x64 images :D
    parser.add_argument("--num_test", default=0.25, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--num_processes",
        help="Use multiprocessing to create dataset.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--only_wt",
        help="Only use whole tumor for training.",
        action="store_true",
    )
    parser.add_argument(
        "--no_corruptions",
        help="Apply artificial corruptions to the test set.",
        action="store_true",
    )
    parser.add_argument(
        "--only_corruptions",
        help="Only apply artificial corruptions to the test set (assume preprocessing has been done before).",
        action="store_true",
    )
    parser.add_argument(
        "--use_default_splits",
        help="Use default splits instead of creating new ones.",
        action="store_true",
    )


def main():
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    num_processes = args.num_processes
    labels = {
        "background": 0,
        "whole_tumor": [1, 2, 3],
        "tumor_core": [2, 3],
        "enhancing_tumor": 3,
    }
    regions_class_order = [1, 2, 3]
    if args.only_wt:
        labels = {
            "background": 0,
            "whole_tumor": 1,
        }
        regions_class_order = None
    meta = {
        "name": "Simple FeTS22 with corruptions",
        "labels": labels,
        "channel_names": {
            "0": "t1",
            "1": "t1ce",
            "2": "t2",
            "3": "flair",
        },
        "dim": 2,
        "file_ending": ".nii.gz",
    }
    if regions_class_order is not None:
        meta["regions_class_order"] = regions_class_order

    target_data_dir = Path(args.out_path)
    target_data_dir.mkdir(exist_ok=True)
    default_split_path = None
    if args.use_default_splits:
        default_split_path = (
            Path(__file__).resolve().parents[4] / "dataset_splits" / "splits_final.json"
        )

    images_tr_dir = target_data_dir / "imagesTr"
    labels_tr_dir = target_data_dir / "labelsTr"
    images_ts_dir = target_data_dir / "imagesTs"
    labels_ts_dir = target_data_dir / "labelsTs"
    if not args.only_corruptions:
        # split cases into train/test
        case_list = [x for x in Path(args.raw_data_path).iterdir() if x.is_dir()]
        if default_split_path is not None:
            all_splits = load_json(default_split_path)
            train_cases = all_splits[0]["train"] + all_splits[0]["val"]
            test_cases = [x for x in case_list if x.name not in train_cases]
            train_cases = [Path(args.raw_data_path) / x for x in train_cases]
            test_cases = [Path(args.raw_data_path) / x for x in test_cases]
        else:
            train_cases, test_cases = train_test_split(
                case_list, test_size=args.num_test, random_state=args.seed
            )
        images_tr_dir.mkdir(parents=True)
        labels_tr_dir.mkdir(parents=True)
        images_ts_dir.mkdir(parents=True)
        labels_ts_dir.mkdir(parents=True)
        meta["numTraining"] = len(train_cases)
        save_json(meta, target_data_dir / "dataset.json")

        case_slice_dict = {}
        logger.info(f"Processing {len(train_cases)} training cases...")
        curr_slice_dict = run_preprocessing_for_case_list(
            args.img_size,
            meta,
            images_tr_dir,
            labels_tr_dir,
            train_cases,
            num_processes=num_processes,
        )
        case_slice_dict.update(curr_slice_dict)
        logger.info(f"Processing {len(test_cases)} test cases...")
        curr_slice_dict = run_preprocessing_for_case_list(
            args.img_size,
            meta,
            images_ts_dir,
            labels_ts_dir,
            test_cases,
            num_processes=num_processes,
        )
        case_slice_dict.update(curr_slice_dict)
        logger.info("Done")
        # save which slice was taken from each case for reference
        save_json(case_slice_dict, target_data_dir / "selected_slices.json")
        # train-val split
        if args.use_default_splits:
            default_split_path = (
                Path(__file__).resolve().parents[4]
                / "dataset_splits"
                / TASK_NAME
                / "splits_final.json"
            )
            all_splits = load_json(default_split_path)
        else:
            all_splits = kfold_split(target_data_dir, num_folds=5, seed=args.seed)
        save_json(all_splits, target_data_dir / "splits_final.json")

    if args.no_corruptions:
        return
    # apply artificial corruptions to the test set
    logger.info("Applying artificial corruptions to the test set...")
    torch.manual_seed(args.seed)
    transform_params, domain_mapping = corrupt_data(
        target_data_dir,
        target_data_dir,
        transform_list=["biasfield", "ghosting", "spike", "affine"],
        magnitude_list=["high"],
        overwrite_ok=True,
    )
    save_json(domain_mapping, target_data_dir / "domain_mapping_00.json")
    save_json(transform_params, target_data_dir / "tio_corruption_params.json")


if __name__ == "__main__":
    main()
