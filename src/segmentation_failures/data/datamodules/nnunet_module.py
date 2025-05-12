import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from loguru import logger
from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from segmentation_failures.data.datamodules.nnunet_utils import (
    MultiThreadedAugmenterWithLength,
    PreprocessImgSegAdapter,
    get_inference_transforms,
    nnUNetTrainerFinite,
    nnUNetTrainerNoDeepSupervisionFinite,
    nnUNetTrainerNoPatchesNoAugNoDeepsup,
    reset_dataloader,
)
from segmentation_failures.utils.data import get_dataset_dir
from segmentation_failures.utils.io import load_json
from segmentation_failures.utils.label_handling import convert_nnunet_regions_to_labels


class NNunetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_id: str,
        fold: int,
        device: str,
        batch_size: int,
        patch_size: list[int],
        nnunet_config: str,
        nnunet_plans_id: str = "nnUNetPlans",
        test_data_root: Optional[str] = None,
        deep_supervision: bool = False,
        num_workers: int | None = None,
        num_workers_preproc: int = 3,  # only for test dataloader
        domain_mapping: int = 0,
        preproc_only: bool = False,  # this can be used to get a train/val dataloader without patching/augmentation
        spacing: list[float] = None,  # this is not used. TODO find a better solution
        legacy_preprocessing=False,
    ) -> None:
        """
        Datamodule wrapper for the nnunet dataloaders and augmentations.

        Args:
            dataset_id: Dataset id.
            fold: Fold id.
            device: Device to use. Can be "cpu", "cuda" or "gpu".
            batch_size: Batch size.
            patch_size: Patch size (shape ZYX). This is included so that I have access to it from the hydra config.
                It is checked against the nnunet configuration.
            nnunet_config: nnunet configuration (e.g. "3d_fullres").
            nnunet_plans_id: nnunet plans id. As in nnunet, it's the name of the folder in the preprocessed data directory.
            test_data_root: Root directory for the test data. Needs to be set for testing.
            deep_supervision: Whether to use deep supervision. The dataloader will return a list of target segmentations on
                different scales if True.
            num_workers: Number of workers for the dataloader/augmenter. If None, let nnunet decide.
                The name was chosen for compatibility with other datamodules, but it's actually n_proc_DA in the nnunet language.
            domain_mapping: Domain mapping id. A corresponding file should be located in the test_data_root directory.
            preproc_only: If True, the dataloader will return the preprocessed data without patching/augmentation.
                Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters()
        self.fold = fold
        self.domain_mapping = domain_mapping
        if isinstance(dataset_id, int):
            dataset_id = str(dataset_id)
        self.dataset_id = dataset_id
        if device == "cpu":
            self.device = torch.device("cpu")
        elif device in ["cuda", "gpu"]:
            self.device = torch.device("cuda")
        if num_workers is not None:
            os.environ["nnUNet_n_proc_DA"] = str(max(1, num_workers))
        self.num_workers_preproc = num_workers_preproc
        # initialize nnunet trainer and use it's dataloaders directly
        self.preproc_data_base = get_dataset_dir(dataset_id, os.environ["nnUNet_preprocessed"])
        if test_data_root is not None:
            try:
                test_data_root = get_dataset_dir(dataset_id, test_data_root)
            except ValueError:
                logger.warning(
                    "No test data directory found. This will result in an error if you use the datamodule for testing."
                )
                test_data_root = None
        self.test_data_base = test_data_root
        self.deep_supervision = deep_supervision
        plans_file = self.preproc_data_base / (nnunet_plans_id + ".json")
        plans = load_json(plans_file)
        dataset_json = load_json(self.preproc_data_base / "dataset.json")
        # So far only default trainer supported.
        # I could make my own class that inherits from nnunet trainer and customize later.
        if preproc_only:
            # deep_supervision is ignored in this case
            trainer_cls = nnUNetTrainerNoPatchesNoAugNoDeepsup
            if deep_supervision:
                logger.info("Ignoring deep_supervision setting for preproc_only=True.")
        elif not deep_supervision:
            trainer_cls = nnUNetTrainerNoDeepSupervisionFinite
        else:
            trainer_cls = nnUNetTrainerFinite
        # I send the nnunet printouts to a temp file because they are not needed
        backup_nnunet_dir = os.environ.get("nnUNet_results")
        os.environ["nnUNet_results"] = tempfile.gettempdir()
        self.nnunet_trainer = trainer_cls(
            plans=plans,
            configuration=nnunet_config,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=True,
            device=self.device,
        )
        os.environ["nnUNet_results"] = backup_nnunet_dir
        # overwriting nnunet config... *sweat*
        if batch_size != self.nnunet_trainer.configuration_manager.batch_size:
            logger.info(
                f"Overwrote batch size value in nnunet config to {batch_size}"
                f" (before:  {self.nnunet_trainer.configuration_manager.batch_size})"
            )
            self.nnunet_trainer.configuration_manager.configuration["batch_size"] = batch_size
            self.nnunet_trainer.batch_size = batch_size
        if not patch_size == self.nnunet_trainer.configuration_manager.patch_size:
            raise ValueError(
                "Patch size mismatch between nnunet config and input. Please synchronize them manually."
            )

        # This is just for convenience, so that I can get the information in my training/testing scripts
        self.preprocess_info = {
            "patch_size": self.nnunet_trainer.configuration_manager.patch_size,
            "spacing": self.nnunet_trainer.configuration_manager.spacing,
            "normalization_schemes": self.nnunet_trainer.configuration_manager.normalization_schemes,
        }
        self.dataset_train = None
        self._dataloader_train = None
        self.dataset_val = None
        self._dataloader_val = None
        self.dataset_test = None
        self._dataloader_test = None

    @property
    def dataset_json(self):
        return self.nnunet_trainer.dataset_json

    def prepare_data(self):
        logger.info("unpacking dataset...")
        unpack_dataset(
            self.nnunet_trainer.preprocessed_dataset_folder,
            unpack_segmentation=True,
            overwrite_existing=False,
            num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
        )
        logger.info("unpacking done...")

    def setup_train(self):
        (
            self._dataloader_train,
            self._dataloader_val,
        ) = self.nnunet_trainer.get_dataloaders()
        if isinstance(self._dataloader_train, SingleThreadedAugmenter):
            self.dataset_train = self._dataloader_train.data_loader._data
            self.dataset_val = self._dataloader_val.data_loader._data
        elif isinstance(self._dataloader_train, MultiThreadedAugmenter):
            self.dataset_train = self._dataloader_train.generator._data
            self.dataset_val = self._dataloader_val.generator._data
        # These dataloaders return batches with keys:
        # 'data', 'properties', 'keys', 'target'

    def setup_test(self):
        if self.test_data_base is None:
            raise ValueError("test_data_root must be specified for testing")
        # need to get the input file and segmentation list
        data_dicts = self.get_test_data_dicts()
        if not self.hparams.legacy_preprocessing:
            data_iter = preprocessing_iterator_fromfiles(
                [x["data"] for x in data_dicts],
                None,
                # loading targets is not easily implemented, because the default nnunet inference doesn't support (GT) labels
                output_filenames_truncated=[
                    x["keys"] for x in data_dicts
                ],  # HACK so that I can get the case ids elsewhere
                plans_manager=self.nnunet_trainer.plans_manager,
                dataset_json=self.nnunet_trainer.dataset_json,
                configuration_manager=self.nnunet_trainer.configuration_manager,
                num_processes=self.num_workers_preproc,
                pin_memory=self.device.type == "cuda",
                verbose=False,
            )
            self._dataloader_test = self._iterator_wrapper_add_batchdim(data_iter)
            self.dataset_test = data_dicts
        else:
            # NOTE I preprocess images and segmentations here and evaluate on the preprocessed segmentations
            ppa = PreprocessImgSegAdapter(
                data_dicts,
                plans_manager=self.nnunet_trainer.plans_manager,
                dataset_json=self.nnunet_trainer.dataset_json,
                configuration_manager=self.nnunet_trainer.configuration_manager,
                num_threads_in_multithreaded=self.num_workers_preproc,
            )
            regions = None
            if self.nnunet_trainer.label_manager.has_regions:
                regions = self.nnunet_trainer.label_manager.foreground_regions
            test_trf = get_inference_transforms(regions)
            if self.num_workers_preproc <= 1:
                mta = SingleThreadedAugmenter(ppa, test_trf)
            else:
                mta = MultiThreadedAugmenterWithLength(
                    ppa,
                    test_trf,
                    self.num_workers_preproc,
                    1,
                    None,
                    pin_memory=self.device.type == "cuda",
                )
            self._dataloader_test = mta
            if hasattr(self._dataloader_test, "data_loader"):
                self.dataset_test = self._dataloader_test.data_loader._data
            elif hasattr(self._dataloader_test, "generator"):
                self.dataset_test = self._dataloader_test.generator._data
            else:
                raise AttributeError(
                    f"Couldn't find the dataset in the dataloader of class {type(self._dataloader_test)}."
                )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ["fit", "validate"] or stage is None:
            self.setup_train()
        elif stage == "test":
            self.setup_test()
        else:
            raise ValueError(f"stage must be fit/test/validate. Got {stage}")

    def train_dataloader(self):
        if self._dataloader_train is None:
            raise RuntimeError(
                "setup() has not been called yet! Please do this before querying dataloaders."
            )
        reset_dataloader(self._dataloader_train)
        return self._dataloader_train

    def val_dataloader(self):
        if self._dataloader_val is None:
            raise RuntimeError(
                "setup() has not been called yet! Please do this before querying dataloaders."
            )
        reset_dataloader(self._dataloader_val)
        return self._dataloader_val

    def test_dataloader(self):
        if self._dataloader_test is None:
            raise RuntimeError(
                "setup() has not been called yet! Please do this before querying dataloaders."
            )
        if not self.hparams.legacy_preprocessing:
            return self._dataloader_test
        reset_dataloader(self._dataloader_test)
        return self._dataloader_test

    def predict_dataloader(self):
        logger.warning(
            "This dataloader is identical to test_dataloader "
            "and was added just for getting rid of a warning."
        )
        return self.test_dataloader()

    def transfer_batch_to_device(
        self, batch: torch.Any, device: torch.device, dataloader_idx: int
    ) -> torch.Any:
        if len(self.hparams.patch_size) == 2:
            # special case: 2D
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        if batch["data"].shape[2:] != self.hparams.patch_size:
            # TODO may not be an ideal condition (what if the image happens to be the same size as the patch size?)
            # I do want to move the data to the device, as I do sliding window inference
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def teardown(self, stage: str) -> None:
        # shut down dataloaders (copied from nnunet trainer)
        old_stdout = sys.stdout
        with open(os.devnull, "w") as f:
            sys.stdout = f
            if self._dataloader_train is not None and isinstance(
                self._dataloader_train, MultiThreadedAugmenter
            ):
                self._dataloader_train._finish()
            if self._dataloader_val is not None and isinstance(
                self._dataloader_val, MultiThreadedAugmenter
            ):
                self._dataloader_val._finish()
            if self._dataloader_test is not None and isinstance(
                self._dataloader_test, MultiThreadedAugmenter
            ):
                self._dataloader_test._finish()
            sys.stdout = old_stdout

    def load_case_id(self, case_id: str, dataset="train", augmentations=True):
        ds = None
        if dataset == "train":
            ds = self.dataset_train
        elif dataset == "valid":
            ds = self.dataset_val
        elif dataset == "test":
            raise NotImplementedError
        else:
            raise ValueError

        if ds is None:
            raise RuntimeError(
                "setup() has not been called yet! Please do this before querying data."
            )
        data, seg, properties = ds.load_case(case_id)
        if augmentations:
            tmp = {
                "data": data,
                "target": seg,
            }
            if dataset == "train":
                transform = self._dataloader_train.transform
            elif dataset == "valid":
                transform = self._dataloader_val.transform
            elif dataset == "test":
                transform = self._dataloader_test.transform
            tmp = transform(**tmp)
            data, seg = tmp["data"], tmp["target"]
        return {
            "keys": case_id,
            "data": data,
            "target": seg,
            "properties": properties,
        }

    def get_test_data_dicts(self):
        img_dir = self.test_data_base / "imagesTs"
        lab_dir = self.test_data_base / "labelsTs"
        domain_mapping_path = (
            self.test_data_base / f"domain_mapping_{self.domain_mapping:02d}.json"
        )
        domain_mapping = None
        if domain_mapping_path.exists():
            domain_mapping = load_json(domain_mapping_path)
        data_dicts = []
        suffix = self.dataset_json.get("file_ending", ".nii.gz")
        for label_path in lab_dir.glob(f"*{suffix}"):
            case_id = label_path.name.removesuffix(suffix)
            img_path_list = []
            num_channels = len(self.nnunet_trainer.dataset_json["channel_names"])
            if "".join(self.nnunet_trainer.dataset_json["channel_names"].values()) == "RGB":
                num_channels = 1  # special case for RGB images (saved in one file)
            for channel_idx in range(num_channels):
                img_path = img_dir / f"{case_id}_{int(channel_idx):04d}{suffix}"
                img_path_list.append(str(img_path))
            curr_dict = {
                "keys": case_id,
                "target": label_path,
                "data": img_path_list,
            }
            if domain_mapping is not None:
                curr_dict["domain_label"] = domain_mapping[case_id]
            data_dicts.append(curr_dict)
        return data_dicts

    def get_val_data_label_paths(self) -> list[str]:
        preproc_dir = Path(self.nnunet_trainer.preprocessed_dataset_folder_base)
        suffix = self.dataset_json.get("file_ending", ".nii.gz")
        gt_files = (preproc_dir / "gt_segmentations").glob(f"*{suffix}")
        split = load_json(preproc_dir / "splits_final.json")
        val_cases = split[self.fold]["val"]
        gt_files = [p for p in gt_files if p.name.removesuffix(suffix) in val_cases]
        assert len(gt_files) == len(val_cases)
        return gt_files

    def regions_to_labels(self, batch_seg: torch.Tensor):
        if self.nnunet_trainer.label_manager.has_regions:
            return convert_nnunet_regions_to_labels(
                batch_seg,
                self.nnunet_trainer.label_manager.regions_class_order,
            )
        return batch_seg

    def _iterator_wrapper_add_batchdim(self, data_iter):
        """This is only used during inference, which uses batch size 1."""
        patch_size = self.hparams.patch_size
        for batch in data_iter:
            if len(patch_size) == 2:
                # special case 2D: remove the z slice dim (CZYX -> CYX)
                data = batch["data"][:, 0]
                # may need to pad (no sliding window used)
                padding = []
                for pdim, idim in zip(patch_size, data.shape[-2:]):
                    padding.extend([(pdim - idim) // 2 + (pdim - idim) % 2, (pdim - idim) // 2])
                batch["data"] = F.pad(data, tuple(padding[::-1]), "constant", 0)
                batch["padding"] = [
                    ((padding[0], padding[1]), (padding[2], padding[3]))
                ]  # list of length batch size (1)
            # add batch dim -> BC[Z]YX
            batch["data"] = batch["data"][None]
            batch["keys"] = [batch["ofile"]]
            batch["properties"] = [batch["data_properties"]]
            del batch["ofile"]
            del batch["data_properties"]
            yield batch

    def post_inference_process(self, pred: torch.Tensor, batch: dict, batch_dim=0) -> torch.Tensor:
        """This method should be used before exporting predictions with nnunet.
        It removes padding and adds the batch dimension for 2D data.

        Args:
            pred: Predictions tensor. Expected shape: [maybe_additional_dims]BC[Z]YX.
            batch: Batch dictionary.
            batch_dim: Batch dimension index. Defaults to 0.
        """
        if batch["data"].ndim == 4:
            if pred.shape[batch_dim] > 1:
                raise NotImplementedError("Inference batch size > 1 not supported.")
            assert batch_dim < pred.ndim - 2
            # 2D case, BCYX
            # remove padding
            padding = batch["padding"][0]
            assert len(padding) == 2
            pred = pred[
                ...,
                padding[0][0] : pred.shape[-2] - padding[0][1],
                padding[1][0] : pred.shape[-1] - padding[1][1],
            ]
            # add dummy z dim
            pred = pred.unsqueeze(-3)
        elif len(batch.get("padding", [])) > 0:
            raise ValueError("Padding is only expected for 2D data.")
        return pred
