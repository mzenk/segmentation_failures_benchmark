import os
from functools import partial
from pathlib import Path
from typing import Optional

import monai
import monai.transforms as trf
import pytorch_lightning as pl
import torch
from loguru import logger
from monai.data import CacheDataset, DataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch.utils.data.sampler import RandomSampler

from segmentation_failures.data.datamodules.nnunet_utils import PreprocessImgSegAdapter
from segmentation_failures.utils.data import get_dataset_dir, make_centered_fg_bbox
from segmentation_failures.utils.io import load_json
from segmentation_failures.utils.label_handling import ConvertSegToRegions

# These are based on the maximum foreground sizes in the training set
# (except 500)
HARDCODED_IMG_SIZES = {
    500: {"crop": (64, 64), "resize_factor": 1},
    503: {"crop": (128, 192, 160), "resize_factor": 1},
    510: {"crop": (24, 128, 128), "resize_factor": 1},
    511: {"crop": (16, 128, 128), "resize_factor": 1},
    514: {"crop": (192, 256, 256), "resize_factor": 0.5},
    515: {"crop": (256, 256, 384), "resize_factor": 0.5},
    520: {"crop": (64, 320, 384), "resize_factor": 1},
    521: {"crop": (24, 160, 160), "resize_factor": 1},
    531: {"crop": (512, 512), "resize_factor": 1},
    540: {"crop": (128, 256, 512), "resize_factor": 0.5},
}


def check_if_files_exist(files):
    if isinstance(files, (list, tuple)):
        for vv in files:
            check_if_files_exist(vv)
    else:
        assert Path(files).exists(), f"File {files} does not exist."


class ClipValueRanged(trf.MapTransform):
    def __init__(
        self,
        keys,
        minv: float = 0.0,
        maxv: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Clips all intensity values to [minv, maxv]
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            minv: Minimum value to clip to.
            maxv: Maximum value to clip to.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.minv = minv
        self.maxv = maxv

    def __call__(self, data: dict[str, torch.Tensor]):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = torch.clamp(d[key], self.minv, self.maxv)
        return d


class VAEdataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_id: str | int,
        fold: int,
        test_data_root: str = None,
        prediction_dir: str = None,
        batch_size: int = 1,
        num_workers: int | None = None,
        num_workers_preproc: int = 4,
        pin_memory: bool = False,
        fixed_steps_per_epoch: Optional[int] = 0,
        domain_mapping: int = 0,
        cache_num: int | float = 0,
        clip_values: tuple = None,
        preprocess_only: bool = False,
        nnunet_configuration: str = "3d_fullres",
        use_metatensor=True,
        fg_center_cropping=True,
        load_images=True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset_id = int(dataset_id)
        if prediction_dir is not None:
            prediction_dir = Path(prediction_dir)
            if not prediction_dir.is_absolute():
                prediction_dir = (
                    get_dataset_dir(dataset_id, os.environ["SEGFAIL_AUXDATA"])
                    / "quality_regression"  # should work also for VAE (preprocessed data)
                    / prediction_dir
                    / "predictions"
                )
        self.prediction_dir = prediction_dir
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.fixed_steps_per_epoch = fixed_steps_per_epoch
        self.cache_num = cache_num
        if isinstance(clip_values, (float, int)):
            clip_values = (-clip_values, clip_values)
        self.clip_values = clip_values
        self.domain_mapping = domain_mapping
        self.nnunet_configuration = nnunet_configuration
        # this is used for the network setup... cf train_image_csf.py
        self.img_size = [
            int(round(x * HARDCODED_IMG_SIZES[self.dataset_id]["resize_factor"]))
            for x in HARDCODED_IMG_SIZES[self.dataset_id]["crop"]
        ]
        self.preproc_only = preprocess_only
        if isinstance(dataset_id, str):
            dataset_id = int(dataset_id)
        if num_workers is None:
            num_workers = get_allowed_n_proc_DA()
        self.num_workers = num_workers  # used for torch dataloader
        self.num_workers_preproc = num_workers_preproc  # used for monai cachedataset
        # hard-coded sh*t, but whatever
        if test_data_root is None:
            test_data_root = os.environ["TESTDATA_ROOT_DIR"]
        self.test_data_root_dir = get_dataset_dir(dataset_id, test_data_root)
        if dataset_id == 500:
            self.train_data_root_dir = get_dataset_dir(dataset_id, os.environ["nnUNet_raw"])
            split_path = self.train_data_root_dir / "splits_final.json"
            self.nnunet_configuration = None
        else:
            self.train_data_root_dir = get_dataset_dir(
                dataset_id, os.environ["nnUNet_preprocessed"]
            )
            split_path = self.train_data_root_dir / "splits_final.json"
        self.crop_foreground = fg_center_cropping
        self.load_images = load_images
        self.train_val_split = load_json(split_path)[fold]
        self.dataset_json = load_json(self.train_data_root_dir / "dataset.json")
        self.dataset_train: CacheDataset = None
        self.dataset_val: CacheDataset = None
        self.dataset_test: CacheDataset = None
        monai.data.meta_obj.set_track_meta(use_metatensor)

    def setup_train(self):
        # set up the correct data path
        train_files, val_files = self.get_train_data_dicts()
        logger.info(f"Found {len(train_files)}/{len(val_files)} cases for training/validation")

        # get data transforms
        train_transforms, val_transforms, _ = get_transforms(
            class_or_regions_dict=self.dataset_json["labels"],
            padcrop_to=HARDCODED_IMG_SIZES[self.dataset_id]["crop"],
            resize_factor=HARDCODED_IMG_SIZES[self.dataset_id]["resize_factor"],
            img_keys="data" if self.load_images else [],
            seg_keys=["target"],
            no_augmentation=self.preproc_only,
            nnunet_dataset=self.dataset_id != 500,
            clip_values=self.clip_values,
            crop_foreground=self.crop_foreground,
        )
        cache_num = self.cache_num
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
        if self.dataset_id == 500:
            train_img_dir = self.train_data_root_dir / "imagesTr"
            train_gt_dir = self.train_data_root_dir / "labelsTr"
        else:
            train_img_dir = self.train_data_root_dir / f"nnUNetPlans_{self.nnunet_configuration}"
            train_gt_dir = train_img_dir
        train_files = []
        val_files = []
        for case_id in self.train_val_split["train"] + self.train_val_split["val"]:
            if self.dataset_id == 500:
                # add label
                label_file = str(train_gt_dir / f"{case_id}.nii.gz")
                # add images
                img_paths = []
                for mod_idx, _ in enumerate(self.dataset_json["channel_names"]):
                    img_paths.append(str(train_img_dir / f"{case_id}_{mod_idx:04d}.nii.gz"))
            else:
                label_file = str(train_gt_dir / f"{case_id}_seg.npy")
                img_paths = str(
                    train_img_dir / f"{case_id}.npy"
                )  # nnunet combines channels during preprocessing
            curr_data_dict = {
                "keys": case_id,
                "data": img_paths,
                "target": label_file,
            }
            for k, v in curr_data_dict.items():
                if k in ["data", "target"] and v is not None:
                    check_if_files_exist(v)
            if case_id in self.train_val_split["train"]:
                train_files.append(curr_data_dict)
            elif case_id in self.train_val_split["val"]:
                val_files.append(curr_data_dict)
            else:
                logger.warning(f"Case {case_id} not found in the split file. Skipping.")
        return train_files, val_files

    def setup_test(self):
        if self.prediction_dir is None:
            raise ValueError("prediction_dir must be set for test dataloader")
        data_dicts = self.get_test_data_dicts()
        _, _, test_transforms = get_transforms(
            class_or_regions_dict=self.dataset_json["labels"],
            padcrop_to=HARDCODED_IMG_SIZES[self.dataset_id]["crop"],
            resize_factor=HARDCODED_IMG_SIZES[self.dataset_id]["resize_factor"],
            img_keys=["data"] if self.load_images else [],
            seg_keys=["pred"],
            no_augmentation=self.preproc_only,
            nnunet_dataset=self.dataset_id != 500,
            clip_values=self.clip_values,
            crop_foreground=self.crop_foreground,
        )
        if self.dataset_id == 500:
            cache_num = self.cache_num
            if isinstance(cache_num, float):
                cache_num = int(cache_num * len(data_dicts))
            self.dataset_test = CacheDataset(
                data=data_dicts,
                transform=test_transforms,
                cache_num=cache_num,
                num_workers=self.num_workers_preproc,
            )
        else:
            # it gets a bit tricky/hacky here, unfortunately.
            # Since the nnunet preprocessor can handle only one segmentation,
            # I drop the GT and pass predictions under the 'target' key
            # (Alternatively, I could modify the preprocessor to accept multiple segs.
            #  While this not difficult, I don't want to mess with it for now.)
            # The VAE model should not get the GT at test-time anyways, so this should be fine.
            # Note that I CANNOT modify the get_test_data_dicts method, because the evaluation callback needs the GT paths.
            pm = PlansManager(self.train_data_root_dir / "nnUNetPlans.json")
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
            cache_num = self.cache_num
            if isinstance(cache_num, float):
                cache_num = int(cache_num * len(data_dicts))
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
            self.test_data_root_dir / f"domain_mapping_{self.domain_mapping:02d}.json"
        )
        domain_mapping = None
        if domain_mapping_path.exists():
            domain_mapping = load_json(domain_mapping_path)
        test_img_dir = self.test_data_root_dir / "imagesTs"
        test_gt_dir = self.test_data_root_dir / "labelsTs"
        suffix = self.dataset_json.get("file_ending", ".nii.gz")
        for label_file in test_gt_dir.glob("*" + suffix):
            case_id = label_file.name.removesuffix(suffix)
            img_paths = []
            for mod_idx, _ in enumerate(self.dataset_json["channel_names"]):
                img_paths.append(str(test_img_dir / f"{case_id}_{mod_idx:04d}{suffix}"))
                if list(self.dataset_json["channel_names"].values()) == ["R", "G", "B"]:
                    # special case
                    break
            # add prediction
            pred_file = list(self.prediction_dir.glob(case_id + ".*"))
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
                "pred": str(pred_file),
                # optional:
                # domain_label
            }
            if domain_mapping is not None:
                curr_data_dict["domain_label"] = domain_mapping[case_id]
            for k, v in curr_data_dict.items():
                if k in ["data", "target", "pred"] and v is not None:
                    check_if_files_exist(v)
            data_dicts.append(curr_data_dict)
        return data_dicts

    def train_dataloader(self):
        replace = self.fixed_steps_per_epoch > 0
        num_samples = len(self.dataset_train)
        if replace:
            num_samples = self.batch_size * self.fixed_steps_per_epoch
        sampler = RandomSampler(
            self.dataset_train,
            replacement=replace,
            num_samples=num_samples,
        )
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            shuffle=False,
            batch_size=self.batch_size if self.dataset_id == 500 else 1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        logger.warning(
            "This dataloader is identical to test_dataloader "
            "and was added just for getting rid of a warning."
        )
        return self.test_dataloader()


def get_transforms(
    class_or_regions_dict: dict,
    padcrop_to,
    resize_factor,
    img_keys: list[str],
    seg_keys: list[str],
    no_augmentation=False,
    nnunet_dataset=True,
    clip_values: tuple = None,
    crop_foreground=False,
):
    img_dim = len(padcrop_to)
    resize_to = [int(round(v * resize_factor)) for v in padcrop_to]
    if img_dim not in [2, 3]:
        raise ValueError("img_dim must be 2 or 3")
    if isinstance(img_keys, str):
        img_keys = [img_keys]
    if isinstance(seg_keys, str):
        seg_keys = [seg_keys]
    # 1. Load transforms
    load_transforms_tr = [
        trf.LoadImaged(
            keys=img_keys + seg_keys,
            ensure_channel_first=False,
            image_only=True,
        ),
    ]
    # if nnunet_dataset:
    #     load_transforms_tr.extend(trf.ToTensord(keys=img_keys + seg_keys))
    # load_transforms_val = load_transforms_tr
    # load_transforms_test = load_transforms_tr
    # if nnunet_dataset:
    #     # preprocessor already loads images as numpy arrays
    #     def squeeze_list(input_list):
    #         if isinstance(input_list, list):
    #             if len(input_list) > 1:
    #                 raise ValueError("input_list must have only one element")
    #             return input_list[0]
    #         return input_list

    #     # nnunet preprocessing outputs BCHWD tensors
    #     load_transforms_test = [
    #         trf.ToTensord(keys=img_keys + seg_keys),
    #         trf.SqueezeDimd(keys=img_keys + seg_keys, dim=0),
    #         # fix for PreprocessImgSegAdapter returning 1-element lists, which are later collated
    #         trf.Lambdad(keys="keys", func=squeeze_list),
    #     ]
    load_transforms_test = load_transforms_tr
    if nnunet_dataset:
        # preprocessor already loads images as numpy arrays
        load_transforms_test = [
            # the properties are currently not used and I got errors during batch collation from it
            trf.DeleteItemsd(keys="properties"),
        ]
        # remove batch dim
        load_transforms_test.append(trf.SqueezeDimd(keys=img_keys + seg_keys, dim=0))
        if len(padcrop_to) == 2:
            # remove z-dim
            load_transforms_tr.append(trf.SqueezeDimd(keys=img_keys + seg_keys, dim=1))
            load_transforms_test.append(trf.SqueezeDimd(keys=img_keys + seg_keys, dim=1))

        def squeeze_list(input_list):
            if isinstance(input_list, list):
                if len(input_list) > 1:
                    raise ValueError("input_list must have only one element")
                return input_list[0]
            return input_list

        # nnunet preprocessing outputs BCHWD tensors
        conversion_trfs = [
            trf.ToTensord(keys=img_keys + seg_keys),
            # fix for PreprocessImgSegAdapter returning 1-element lists, which are later collated
            trf.Lambdad(keys="keys", func=squeeze_list),
        ]
        load_transforms_tr.extend(conversion_trfs)
        load_transforms_test.extend(conversion_trfs)
    # Images will have shape C, H, W[, D]
    # 2. Preprocessing transforms
    resize_trfs_tr = resize_trfs_ts = [
        trf.ResizeWithPadOrCropd(
            keys=img_keys + seg_keys, spatial_size=padcrop_to, mode="constant"
        ),
    ]
    if crop_foreground:
        resize_trfs_tr = [
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
        resize_trfs_ts = [
            trf.CropForegroundd(
                keys=img_keys + seg_keys,
                source_key="pred",  # !
                margin=0,
                select_fn=partial(make_centered_fg_bbox, bbox_size=padcrop_to),
            ),
        ]
        resize_trfs_ts.extend(resize_trfs_tr[1:])

    preproc_transforms = [
        ConvertSegToRegions(
            seg_keys, class_or_regions_dict, include_background=False
        ),  # if there are no regions, this converts to one-hot
        trf.EnsureTyped(keys=seg_keys, dtype=torch.int16),
    ]
    if len(img_keys) > 0:
        preproc_transforms += [trf.EnsureTyped(keys=img_keys, dtype=torch.float32)]

    if not nnunet_dataset and len(img_keys) > 0:
        # nnunet preprocessing (offline) already normalizes the images
        preproc_transforms.append(
            trf.NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True)
        )

    # 3. augmentation transforms
    augmentation = []
    if not no_augmentation:
        augmentation = [
            trf.RandZoomd(
                keys=img_keys + seg_keys,
                min_zoom=0.9,
                max_zoom=1.1,
                mode=["trilinear" if img_dim == 3 else "bilinear"] * len(img_keys)
                + ["nearest"] * len(seg_keys),
                align_corners=[True] * len(img_keys) + [None] * len(seg_keys),
                prob=0.2,
            ),
            *[
                trf.RandFlipd(img_keys + seg_keys, spatial_axis=[i], prob=0.5)
                for i in range(img_dim)
            ],
        ]
        if nnunet_dataset:
            # TODO dirty; for dataset 500, I use affine transforms as test-time corruptions
            # so I don't want to use them for augmentation
            augmentation.append(
                trf.RandAffined(
                    keys=img_keys + seg_keys,
                    mode=["trilinear" if img_dim == 3 else "bilinear"] * len(img_keys)
                    + ["nearest"] * len(seg_keys),
                    rotate_range=(0.15, 0, 0),
                    scale_range=None,  # zoomed above
                    shear_range=None,
                    translate_range=(5, 5, 5),
                    padding_mode="border",
                    prob=0.2,
                )
            )
        if len(img_keys) > 0:
            augmentation += [
                trf.RandScaleIntensityd(keys=img_keys, factors=0.1, prob=0.5),
                trf.RandGaussianNoised(keys=img_keys, std=0.05, prob=0.2),
            ]
    clip_trf = []
    if clip_values is not None and len(img_keys) > 0:
        clip_trf = [ClipValueRanged(keys=img_keys, minv=clip_values[0], maxv=clip_values[1])]
    train_transform = trf.Compose(
        load_transforms_tr + resize_trfs_tr + preproc_transforms + augmentation + clip_trf
    )
    val_transform = trf.Compose(
        load_transforms_tr + resize_trfs_tr + preproc_transforms + clip_trf
    )
    # TODO The validation dataloader is problematic (resize transform). I have two validation methods
    # (VAE and linear regression of Dice) and they should use training/testing style resizing.
    test_transform = trf.Compose(
        load_transforms_test + resize_trfs_ts + preproc_transforms + clip_trf
    )
    return train_transform, val_transform, test_transform
