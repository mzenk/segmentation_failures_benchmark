"""
What this module does:

- provide a dataloader for the quality regression data, which consists of (image, segmentation, groundtruth) tuples
- optionally, also include the pixel-wise confidence in the dataloader
- quality targets are computed from the segmentation and the ground truth segmentation on the fly
- augmentations are possible but turned off by default
"""

import os
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import monai
import monai.transforms as trf
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from monai.data import CacheDataset, DataLoader
from monai.utils import Method, PytorchPadMode
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from segmentation_failures.data.datamodules.nnunet_utils import PreprocessImgSegAdapter
from segmentation_failures.evaluation.segmentation.segmentation_metrics import (
    get_metrics_and_info,
)
from segmentation_failures.utils.data import get_dataset_dir, make_centered_fg_bbox
from segmentation_failures.utils.io import load_json
from segmentation_failures.utils.label_handling import (
    ConvertSegToRegions,
    convert_to_onehot,
)

# These are chosen to cover all of the foreground for all training cases
# and the whole image for most training cases (~75%).
# Resizing is applied to train on 11GB GPU.
HARDCODED_IMG_SIZES = {
    500: {"crop": (64, 64), "resize_factor": 1},
    503: {"crop": (160, 192, 160), "resize_factor": 1},
    510: {"crop": (24, 320, 320), "resize_factor": 1},
    511: {"crop": (16, 320, 320), "resize_factor": 1},
    514: {"crop": (192, 256, 256), "resize_factor": 0.5},
    515: {"crop": (256, 448, 448), "resize_factor": 0.5},
    520: {"crop": (96, 512, 512), "resize_factor": 0.5},
    521: {"crop": (24, 320, 320), "resize_factor": 1},
    531: {"crop": (512, 512), "resize_factor": 1},
    540: {"crop": (128, 512, 512), "resize_factor": 0.5},
}


def check_if_files_exist(files):
    if isinstance(files, (list, tuple)):
        for vv in files:
            check_if_files_exist(vv)
    else:
        assert Path(files).exists(), f"File {files} does not exist."


class QualityRegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_id: int,
        fold: int,
        prediction_dir: str,  # if an absolute path is given, it is used directly; else, search in $SEGFAIL_AUXDATA
        metric_targets: str | list[str],
        test_data_root: str = None,
        confid_dir: str = None,
        confid_name: str = None,
        batch_size: int = 1,
        num_workers: int | None = None,
        num_workers_preproc: int = 4,
        pin_memory: bool = False,
        domain_mapping: int = 0,
        preproc_only: bool = False,  # this can be used to get a train dataloader without augmentation
        cache_num: int | float = 0,
        nnunet_configuration: str = "3d_fullres",
        use_metatensor=True,
        randomize_prediction: float = 0.0,
        include_background=False,  # both in the target passed to the network and in the seg metrics
        expt_group="default",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.nnunet_configuration = nnunet_configuration
        if isinstance(metric_targets, str):
            metric_targets = [metric_targets]
        if metric_targets is None:
            # not strictly necessary for testing
            self.metric_targets, metric_infos = None, {}
        else:
            self.metric_targets, metric_infos = get_metrics_and_info(
                metric_targets,
                include_background=True,
                # include bg here because it is filtered in the transforms (ConvertSegToRegions)
            )
        self.domain_mapping = domain_mapping
        self.preproc_only = preproc_only
        if isinstance(dataset_id, str):
            dataset_id = int(dataset_id)
        self.dataset_id = dataset_id
        if num_workers is None:
            num_workers = get_allowed_n_proc_DA()
        self.num_workers = num_workers  # used for torch dataloader
        self.num_workers_preproc = num_workers_preproc  # used for monai cachedataset
        # hard-coded sh*t, but whatever
        if dataset_id == 500:
            self.train_data_root = get_dataset_dir(dataset_id, os.environ["nnUNet_raw"])
            split_path = self.train_data_root / "splits_final.json"
            orig_spacing = [1.0, 1.0]
        else:
            self.train_data_root = get_dataset_dir(dataset_id, os.environ["nnUNet_preprocessed"])
            split_path = self.train_data_root / "splits_final.json"
            orig_spacing = (
                PlansManager(self.train_data_root / "nnUNetPlans.json")
                .get_configuration(nnunet_configuration)
                .spacing
            )
        if test_data_root is None:
            test_data_root = os.environ["TESTDATA_ROOT_DIR"]
        self.test_data_root = get_dataset_dir(dataset_id, test_data_root)
        self.train_val_split = load_json(split_path)[fold]
        if Path(prediction_dir).is_absolute():
            self.prediction_dir = Path(prediction_dir)
        else:
            self.prediction_dir = (
                get_dataset_dir(dataset_id, os.environ["SEGFAIL_AUXDATA"])
                / expt_group
                / "quality_regression"
                / prediction_dir
                / "predictions"
            )
        self.confid_dir = None
        self.confid_name = confid_name
        if confid_name is not None:
            assert confid_dir is not None
            if Path(confid_dir).is_absolute():
                self.confid_dir = Path(confid_dir)
            else:
                self.confid_dir = (
                    get_dataset_dir(dataset_id, os.environ["SEGFAIL_AUXDATA"])
                    / "quality_regression"
                    / confid_dir
                    / "confidence_maps"
                )
        self.dataset_json = load_json(self.train_data_root / "dataset.json")
        self.metric_target_names = []
        self.metric_higher_better = []
        for metric_name, metric_info in metric_infos.items():
            if metric_info.classwise:
                num_classes = len(self.dataset_json["labels"]) - (not include_background)
                self.metric_target_names += [f"{metric_name}_{i}" for i in range(num_classes)]
                self.metric_higher_better += [metric_info.higher_better] * num_classes
            else:
                self.metric_target_names.append(metric_name)
                self.metric_higher_better.append(metric_info.higher_better)
        self.dataset_train: CacheDataset = None
        self._dataloader_train = None
        self.dataset_val: CacheDataset = None
        self._dataloader_val = None
        self.dataset_test: CacheDataset = None
        self._dataloader_test = None

        self.dataset_fingerprint = {
            "spacing": [
                s / HARDCODED_IMG_SIZES[self.dataset_id]["resize_factor"] for s in orig_spacing
            ],
            "img_size": [
                int(round(x * HARDCODED_IMG_SIZES[self.dataset_id]["resize_factor"]))
                for x in HARDCODED_IMG_SIZES[self.dataset_id]["crop"]
            ],
        }  # this can be used for configuring the network
        monai.data.meta_obj.set_track_meta(use_metatensor)

    def setup_train(self):
        # set up the correct data path
        train_files, val_files = self.get_train_data_dicts()
        logger.info(f"Found {len(train_files)}/{len(val_files)} cases for training/validation")

        # get data transforms
        train_transforms, val_transforms, _ = get_transforms(
            self.dataset_json["labels"],
            seg_metrics=self.metric_targets,
            padcrop_to=HARDCODED_IMG_SIZES[self.dataset_id]["crop"],
            resize_factor=HARDCODED_IMG_SIZES[self.dataset_id]["resize_factor"],
            confid_keys=["confid"] if self.confid_dir else None,
            no_augmentation=self.preproc_only,
            randomize_prediction=self.hparams.randomize_prediction,
            is_nnunet_preprocessed=self.dataset_id != 500,
            include_background=self.hparams.include_background,
        )
        cache_num = self.hparams.cache_num
        if isinstance(cache_num, float):
            cache_num = cache_num * (len(train_files) + len(val_files))
        # not sure what is the optimal split (wrt training runtime); could also cache only train samples
        cache_num_train = int(cache_num * len(train_files) / (len(train_files) + len(val_files)))
        cache_num_valid = int(cache_num * len(val_files) / (len(train_files) + len(val_files)))
        self.dataset_train = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_num=cache_num_train,
            num_workers=self.num_workers_preproc,
        )
        self.dataset_val = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_num=cache_num_valid,
            num_workers=self.num_workers_preproc,
        )

    def get_train_data_dicts(self):
        pred_to_case_id_mapping = load_json(self.prediction_dir / "prediction_to_case_id.json")
        if self.dataset_id == 500:
            train_img_dir = self.train_data_root / "imagesTr"
            train_gt_dir = self.train_data_root / "labelsTr"
            pred_file_generator = self.prediction_dir.glob("*.nii.gz")
        else:
            train_img_dir = self.train_data_root / f"nnUNetPlans_{self.nnunet_configuration}"
            train_gt_dir = train_img_dir
            pred_file_generator = self.prediction_dir.glob("*.npy")
        train_files = []
        val_files = []

        for pred_file in pred_file_generator:
            case_id = pred_to_case_id_mapping[pred_file.name]
            if self.dataset_id == 500:
                label_file = str(train_gt_dir / f"{case_id}.nii.gz")
                img_paths = []
                for mod_idx, _ in enumerate(self.dataset_json["channel_names"]):
                    img_paths.append(str(train_img_dir / f"{case_id}_{mod_idx:04d}.nii.gz"))
            else:
                label_file = str(train_gt_dir / f"{case_id}_seg.npy")
                img_paths = str(
                    train_img_dir / f"{case_id}.npy"
                )  # nnunet combines channels during preprocessing
            # add confidence
            curr_data_dict = {
                "keys": case_id,
                "data": img_paths,
                "target": label_file,
                "pred": str(pred_file),
                # optional:
                # confid
            }
            if self.confid_dir is not None:
                file_ending = ".nii.gz" if self.dataset_id == 500 else ".npy"
                curr_data_dict["confid"] = str(
                    self.confid_dir
                    / f"{pred_file.name.split('.')[0]}_csf={self.confid_name}{file_ending}"
                )
            for k, v in curr_data_dict.items():
                if k in ["data", "target", "pred", "confid"] and v is not None:
                    check_if_files_exist(v)
            if case_id in self.train_val_split["train"]:
                train_files.append(curr_data_dict)
            elif case_id in self.train_val_split["val"]:
                val_files.append(curr_data_dict)
            else:
                logger.warning(f"Case {case_id} not found in the split file. Skipping.")
        return train_files, val_files

    def setup_test(self):
        data_dicts = self.get_test_data_dicts()
        _, _, test_transforms = get_transforms(
            self.dataset_json["labels"],
            padcrop_to=HARDCODED_IMG_SIZES[self.dataset_id]["crop"],
            resize_factor=HARDCODED_IMG_SIZES[self.dataset_id]["resize_factor"],
            seg_metrics=self.metric_targets,
            seg_keys=["pred"],  # no target at test time for this method
            confid_keys=["confid"] if self.confid_dir else None,
            no_augmentation=self.preproc_only,
            is_nnunet_preprocessed=self.dataset_id != 500,
            include_background=self.hparams.include_background,
        )
        cache_num = self.hparams.cache_num
        if isinstance(cache_num, float):
            cache_num = int(cache_num * len(data_dicts))
        if self.dataset_id == 500:
            self.dataset_test = CacheDataset(
                data=data_dicts,
                transform=test_transforms,
                cache_num=cache_num,
                num_workers=self.num_workers_preproc,
            )
        else:
            if self.confid_dir is not None:
                # probably it's easiest to just save the "preprocessed" confidence maps, too. -> confidence writer
                raise NotImplementedError
            # it gets a bit tricky/hacky here, unfortunately.
            # Since the nnunet preprocessor can handle only one segmentation,
            # I drop the GT and pass predictions under the 'target' key
            # (Alternatively, I could modify the preprocessor to accept multiple segs.
            #  While this not difficult, I don't want to mess with it for now.)
            # The QR model should not get the GT at test-time anyways, so this should be fine.
            # Note that I CANNOT modify the get_test_data_dicts method, because the evaluation callback needs the GT paths.
            pm = PlansManager(self.train_data_root / "nnUNetPlans.json")
            config_manager = pm.get_configuration(self.nnunet_configuration)
            preprocessing_adapter = PreprocessImgSegAdapter(
                data_dicts=data_dicts,
                seg_key="pred",
                plans_manager=pm,
                dataset_json=self.dataset_json,
                configuration_manager=config_manager,
                num_threads_in_multithreaded=1,
                # using 1 above is important, because the torch-based dataloader works
                # differently from the nnunet dataloader.
            )
            self.dataset_test = CacheDataset(
                preprocessing_adapter,
                transform=test_transforms,
                cache_num=cache_num,
                num_workers=self.num_workers_preproc,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ["fit", "validate"] or stage is None:
            self.setup_train()
        elif stage == "test":
            self.setup_test()
        else:
            raise ValueError(f"stage must be fit/test/validate. Got {stage}")

    def get_test_data_dicts(self):
        data_dicts = []
        domain_mapping_path = (
            self.test_data_root / f"domain_mapping_{self.domain_mapping:02d}.json"
        )
        domain_mapping = None
        if domain_mapping_path.exists():
            domain_mapping = load_json(domain_mapping_path)
        test_img_dir = self.test_data_root / "imagesTs"
        test_gt_dir = self.test_data_root / "labelsTs"
        suffix = self.dataset_json.get("file_ending", ".nii.gz")
        for label_file in test_gt_dir.glob(f"*{suffix}"):
            case_id = label_file.name.removesuffix(suffix)
            # add images
            img_paths = []
            for mod_idx, _ in enumerate(self.dataset_json["channel_names"]):
                img_paths.append(str(test_img_dir / f"{case_id}_{mod_idx:04d}{suffix}"))
                if list(self.dataset_json["channel_names"].values()) == ["R", "G", "B"]:
                    # special case
                    break
            # add prediction
            pred_file = list(self.prediction_dir.glob(f"{case_id}.*"))
            if len(pred_file) == 0:
                logger.warning(f"Found no predictions for case {case_id}")
                continue
            elif len(pred_file) == 1:
                pred_file = str(pred_file[0])
            else:
                raise ValueError(f"Found {len(pred_file)} predictions for case {case_id}")
            curr_data_dict = {
                "keys": case_id,
                "data": img_paths,
                "target": str(label_file),
                "pred": pred_file,
                # optional:
                # confid
                # domain_label
            }
            if self.confid_dir is not None:
                curr_data_dict["confid"] = str(
                    self.confid_dir / f"{case_id}_csf={self.confid_name}{suffix}"
                )
            if domain_mapping is not None:
                curr_data_dict["domain_label"] = domain_mapping[case_id]
            for k, v in curr_data_dict.items():
                if k in ["data", "target", "pred", "confid"] and v is not None:
                    check_if_files_exist(v)
            data_dicts.append(curr_data_dict)
        return data_dicts

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self):
        logger.warning(
            "This dataloader is identical to test_dataloader "
            "and was added just for getting rid of a warning."
        )
        return self.test_dataloader()


def get_transforms(
    class_or_regions_dict: dict,
    seg_metrics: dict[str, monai.metrics.Metric],
    padcrop_to,
    resize_factor,
    img_keys: list[str] = None,
    seg_keys: list[str] = None,
    confid_keys: list[str] = None,
    no_augmentation=False,
    randomize_prediction=0.25,
    is_nnunet_preprocessed=True,
    include_background=False,
):
    resize_to = [int(round(p * resize_factor)) for p in padcrop_to]
    img_dim = len(padcrop_to)
    if img_dim not in [2, 3]:
        raise ValueError("img_dim must be 2 or 3")
    if img_keys is None:
        img_keys = ["data"]
    if seg_keys is None:
        seg_keys = ["target", "pred"]
    if confid_keys is None:
        confid_keys = []
    # 1. load transforms
    load_transforms_tr = [
        trf.LoadImaged(
            keys=img_keys + confid_keys + seg_keys,
            ensure_channel_first=False,
            image_only=True,
        ),
    ]
    load_transforms_test = load_transforms_tr
    if is_nnunet_preprocessed:
        # preprocessor already loads images as numpy arrays; shape BCHWD
        load_transforms_test = [
            # the properties are currently not used and I got errors during batch collation from it
            trf.DeleteItemsd(keys="properties"),
        ]
        # remove batch dim
        load_transforms_test.append(trf.SqueezeDimd(keys=img_keys + confid_keys + seg_keys, dim=0))
        if len(padcrop_to) == 2:
            # remove z-dim
            load_transforms_tr.append(
                trf.SqueezeDimd(keys=img_keys + confid_keys + seg_keys, dim=1)
            )
            load_transforms_test.append(
                trf.SqueezeDimd(keys=img_keys + confid_keys + seg_keys, dim=1)
            )

        def squeeze_list(input_list):
            if isinstance(input_list, list):
                if len(input_list) > 1:
                    raise ValueError("input_list must have only one element")
                return input_list[0]
            return input_list

        # nnunet preprocessing outputs BCHWD tensors
        conversion_trfs = [
            trf.ToTensord(keys=img_keys + confid_keys + seg_keys),
            # fix for PreprocessImgSegAdapter returning 1-element lists, which are later collated
            trf.Lambdad(keys="keys", func=squeeze_list),
        ]
        load_transforms_tr.extend(conversion_trfs)
        load_transforms_test.extend(conversion_trfs)
    # 2. Preprocessing transforms
    preproc_transforms_tr = [
        # this crops with padcrop_to around the foreground region
        # if the fg is larger, it crops to the fg size
        trf.CropForegroundd(
            keys=img_keys + seg_keys,
            source_key="target",
            margin=0,
            select_fn=partial(make_centered_fg_bbox, bbox_size=padcrop_to),
        ),
        # only pads if the image is smaller than the crop size
        trf.SpatialPadd(keys=img_keys + seg_keys, spatial_size=padcrop_to),
        # two scenarios in which this is used
        # a) resize_factor != 1
        # b) resize_factor == 1 and fg size > padcrop_to => distorts the image but ignore for now (should be rare)
        trf.Resized(
            keys=img_keys + seg_keys,
            spatial_size=resize_to,
            mode=["trilinear" if img_dim == 3 else "bilinear"] * len(img_keys)
            + ["nearest"] * len(seg_keys),
        ),
    ]
    # can't crop based on GT during testing
    preproc_transforms_ts = [
        trf.CropForegroundd(
            keys=img_keys + seg_keys,
            source_key="pred",  # !
            margin=0,
            select_fn=partial(make_centered_fg_bbox, bbox_size=padcrop_to),
        ),
    ]
    if len(confid_keys) > 0:
        logger.warning("Quality regression with confidence maps is experimental.")
        preproc_transforms_tr += [
            ResizeWithPadOrCropMaxValued(
                keys=confid_keys, spatial_size=padcrop_to, mode="constant"
            )
        ]
    preproc_transforms_tr += [
        ConvertSegToRegions(
            seg_keys,
            class_or_regions_dict,
            include_background=include_background,
        ),  # if there are no regions, this converts to one-hot
    ]
    if not is_nnunet_preprocessed:
        # nnunet preprocessing (offline) already normalizes the images
        preproc_transforms_tr.append(
            trf.NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True)
        )
        # TODO normalize confidences?
    # Add remaining preprocessing transforms to test transforms
    preproc_transforms_ts.extend(preproc_transforms_tr[1:])

    # 3. spatial transforms
    augmentation = []
    if not no_augmentation:
        # if confidence maps are included, I'm cautious with augmentation
        if len(confid_keys) == 0:
            augmentation = [
                trf.RandZoomd(
                    keys=img_keys + seg_keys,
                    min_zoom=0.9,
                    max_zoom=1.1,
                    mode=["trilinear" if img_dim == 3 else "bilinear"] * len(img_keys)
                    + ["nearest"] * len(seg_keys),
                    align_corners=[True] * len(img_keys) + [None] * len(seg_keys),
                    prob=0.15,
                )
            ]
        augmentation.extend(
            [
                trf.RandGaussianNoised(keys=img_keys, std=0.01, prob=0.15),
                trf.RandScaleIntensityd(keys=img_keys, factors=0.3, prob=0.15),
                *[
                    trf.RandFlipd(img_keys + confid_keys + seg_keys, spatial_axis=[i], prob=0.5)
                    for i in range(img_dim)
                ],
            ]
        )
        if len(confid_keys) == 0 and randomize_prediction > 0:
            # elastic_trf = trf.Rand3DElasticd if img_dim == 3 else trf.Rand2DElasticd
            # augmentation.append(
            #     elastic_trf(
            #         keys=["pred"],
            #         prob=0.25,
            #         rotate_range=(0.25, 0., 0.),
            #         scale_range=(0.85, 1.25),
            #         translate_range=(2, 20, 20),
            #         mode="nearest",
            #         padding_mode="zeros",
            #     )
            # )
            augmentation.append(
                trf.RandAffined(
                    keys=["pred"],
                    prob=randomize_prediction,
                    rotate_range=(0.26, 0.0, 0.0)[-img_dim:],
                    scale_range=(0.2, 0.2, 0.2)[-img_dim:],
                    translate_range=(1, 10, 10)[-img_dim:],
                    mode="nearest",
                    padding_mode="zeros",
                )
            )
    metric_target_computation = [
        # I could also compute metrics in the lightning module
        # For now I do it here
        SegMetricTargetComputation("pred", "target", seg_metrics),
        # trf.DeleteItemsd(keys="target"),  # not needed anymore, but I keep it for the batch visualization
        trf.CastToTyped(keys="target", dtype=torch.uint8),
    ]
    train_transform = trf.Compose(
        load_transforms_tr + preproc_transforms_tr + augmentation + metric_target_computation
    )
    val_transform = trf.Compose(
        load_transforms_tr + preproc_transforms_tr + metric_target_computation
    )
    test_transform = trf.Compose(load_transforms_test + preproc_transforms_ts)
    return train_transform, val_transform, test_transform


class SegMetricTargetComputation(trf.MapTransform):
    def __init__(
        self, pred_key, target_key, metric_objects: dict, labels_or_regions_defs: dict = None
    ):
        # TODO the metrics interface is not explicit here; I use MONAI metrics
        super().__init__([pred_key, target_key])
        self.pred_key = pred_key
        self.target_key = target_key
        self.metric_objs = metric_objects
        self.labels_or_regions_defs = labels_or_regions_defs
        if labels_or_regions_defs is not None:
            self.labels_or_regions_defs = list(labels_or_regions_defs.values())

    def __call__(self, data):
        target_arr = []
        # segmentation shapes: (C, H, W, D)
        pred = data[self.pred_key]
        target = data[self.target_key]
        if self.labels_or_regions_defs is not None:
            pred = convert_to_onehot(pred, self.labels_or_regions_defs)
            target = convert_to_onehot(target, self.labels_or_regions_defs)
        for metric_obj in self.metric_objs.values():
            metric_obj.reset()  # annoying; metrics currently have buffer that I don't want to use here
            # compute metric between pred and target. Unsqueeze because metric expects batch dimension
            metric_val = metric_obj(pred.unsqueeze(0), target.unsqueeze(0))
            if len(metric_val.shape) == 2:
                metric_val = metric_val.squeeze(0)
            target_arr.append(metric_val)
        data["metric_target"] = torch.cat(target_arr, dim=0)
        return data


def compute_padded_size(input_size: list[int], max_size_after_downsampling=8):
    # this is used for getting a shape that is compatible with the dynunet, which downsamples until
    # each spatial size is < 8. This is probably not the fastest solution, but I don't know how to do it better atm.
    candidates = []
    input_size = np.array(input_size)
    for m in range(1, max_size_after_downsampling + 1):
        if m % 2 == 0:
            continue
        # find k such that m * 2**k >= input_size and m * 2**(k-1) < input_size
        best_k = np.floor(np.log2(input_size / m))
        best_k[m * 2**best_k < input_size] += 1
        candidates.append(m * 2**best_k)
    candidates = np.array(candidates).astype(int)
    diffs = candidates - input_size.reshape(1, -1)
    assert np.all(diffs >= 0)
    return candidates[diffs.argmin(axis=0), np.arange(candidates.shape[1])].tolist()


class DynamicSpatialPad(trf.SpatialPad):
    def __init__(
        self,
        max_size_after_downsampling: int,
        method: str = Method.SYMMETRIC,
        mode: str = PytorchPadMode.CONSTANT,
        **kwargs,
    ) -> None:
        super().__init__(None, method, mode, **kwargs)
        self.max_size_after_downsampling = max_size_after_downsampling

    def compute_pad_width(self, spatial_shape: Sequence[int]) -> List[Tuple[int, int]]:
        """
        dynamically compute the pad width according to the spatial shape.

        Args:
            spatial_shape: spatial shape of the original image.

        """
        new_size = compute_padded_size(spatial_shape, self.max_size_after_downsampling)
        if self.method == Method.SYMMETRIC:
            pad_width = []
            for i, sp_i in enumerate(new_size):
                width = max(sp_i - spatial_shape[i], 0)
                pad_width.append((width // 2, width - (width // 2)))
        else:
            pad_width = [(0, max(sp_i - spatial_shape[i], 0)) for i, sp_i in enumerate(new_size)]
        return [(0, 0)] + pad_width


class DynamicSpatialPadd(trf.Padd):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialPad`.
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.

    """

    def __init__(
        self,
        keys,
        max_size_after_downsampling: int,
        method=Method.SYMMETRIC,
        mode=PytorchPadMode.CONSTANT,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        """
        Same as the SpatialPadd transform, but with dynamic padding width.
        """
        padder = DynamicSpatialPad(max_size_after_downsampling, method, **kwargs)
        super().__init__(keys, padder=padder, mode=mode, allow_missing_keys=allow_missing_keys)


class ResizeWithPadOrCropMaxValue(trf.ResizeWithPadOrCrop):
    """
    Minor modification of ResizeWithPadOrCrop that pads with the maximum value instead of a fixed value.
    """

    def __init__(
        self,
        spatial_size: Sequence[int] | int,
        method: str = Method.SYMMETRIC,
        mode: str = PytorchPadMode.CONSTANT,
        lazy: bool = False,
        **pad_kwargs,
    ):
        if mode != PytorchPadMode.CONSTANT:
            raise ValueError("Only constant padding is supported")
        super().__init__(spatial_size, method, mode, lazy, **pad_kwargs)

    def __call__(  # type: ignore[override]
        self, img: torch.Tensor, mode: str | None = None, lazy: bool | None = None, **pad_kwargs
    ) -> torch.Tensor:
        if mode != PytorchPadMode.CONSTANT:
            raise ValueError("Only constant padding is supported")
        pad_kwargs.update({"value": img.max()})
        return super().__call__(img, mode, lazy, **pad_kwargs)


class ResizeWithPadOrCropMaxValued(trf.Padd):
    """
    Minor modification of ResizeWithPadOrCrop that pads with the maximum value instead of a fixed value.
    """

    def __init__(
        self,
        keys,
        spatial_size: Sequence[int] | int,
        mode,
        allow_missing_keys: bool = False,
        method: str = Method.SYMMETRIC,
        lazy: bool = False,
        **pad_kwargs,
    ) -> None:
        padcropper = ResizeWithPadOrCropMaxValue(
            spatial_size=spatial_size, method=method, **pad_kwargs, lazy=lazy
        )
        super().__init__(
            keys, padder=padcropper, mode=mode, allow_missing_keys=allow_missing_keys, lazy=lazy  # type: ignore
        )
