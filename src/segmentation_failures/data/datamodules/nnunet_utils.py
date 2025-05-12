from time import sleep, time
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
    RemoveLabelTransform,
)
from loguru import logger
from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import (
    ConvertSegmentationToRegionsTransform,
)
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoDA import (
    nnUNetTrainerNoDA,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


class SimpleDataLoader(DataLoader):
    def __init__(
        self,
        data: nnUNetDataset,
        batch_size: int,
        patch_size: list[int],
        label_manager: LabelManager,
        num_threads_in_multithreaded: int = 1,
        shuffle=True,
        seed=None,
        return_incomplete=True,
        infinite=False,
        sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
    ):
        super().__init__(
            data,
            batch_size=batch_size,
            num_threads_in_multithreaded=num_threads_in_multithreaded,
            seed_for_shuffle=seed,
            return_incomplete=return_incomplete,
            shuffle=shuffle,
            infinite=infinite,
            sampling_probabilities=sampling_probabilities,
        )
        assert isinstance(
            data, nnUNetDataset
        ), "nnUNetDataLoaderBase only supports dictionaries as data"
        self.indices = list(data.keys())
        self.num_channels = None
        self.annotated_classes_key = tuple(label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.patch_size = patch_size

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        case_properties = []
        data_all = []
        seg_all = []
        padding_all = []
        for data_idx in selected_keys:
            data, seg, properties = self._data.load_case(data_idx)
            if len(self.patch_size) == 2:  # 2D case, C1HW
                # remove zdim (1)
                assert data.shape[1] == 1, "2D data should have 1 z-slice"
                assert seg.shape[1] == 1, "2D data should have 1 z-slice"
                data = data[:, 0]
                seg = seg[:, 0]
                # need to pad to patch size
                padding = [
                    ((pdim - idim) // 2 + (pdim - idim) % 2, (pdim - idim) // 2)
                    for pdim, idim in zip(self.patch_size, data.shape[1:])
                ]
                padding_all.append(padding)
                data = np.pad(data, [(0, 0), *padding], "constant", constant_values=0)
                seg = np.pad(seg, [(0, 0), *padding], "constant", constant_values=-1)
            data_all.append(data)
            seg_all.append(seg)
            case_properties.append(properties)
        data_all = np.stack(data_all, axis=0)
        seg_all = np.stack(seg_all, axis=0)
        return {
            "data": data_all,
            "seg": seg_all,
            "properties": case_properties,
            "keys": selected_keys,
            "padding": padding_all,
        }


class MultiThreadedAugmenterWithLength(MultiThreadedAugmenter):
    # # TODO something in this class is interfering with the lightning training loop
    # (every second epoch is skipped because the Dataloaders raise a StopIteration right away)
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.length = int(np.ceil(len(self.generator.indices) / self.generator.batch_size).item())

    # def __len__(self):
    #     return self.length
    # TODO this is a workaround for issues that prevent the testing pipeline to exit.
    # In the long run, I want to avoid calling .kill on the processes
    def _finish(self, timeout=10):
        self.abort_event.set()

        start = time()
        while (
            self.pin_memory_thread is not None
            and self.pin_memory_thread.is_alive()
            and start + timeout > time()
        ):
            sleep(0.2)

        if len(self._processes) != 0:
            logger.debug("MultiThreadedGenerator: shutting down workers...")
            [i.kill() for i in self._processes]

            for i, _ in enumerate(self._processes):
                self._queues[i].close()
                self._queues[i].join_thread()

            self._queues = []
            self._processes = []
            self._queue = None
            self._end_ctr = 0
            self._queue_ctr = 0

            del self.pin_memory_queue
        self.was_initialized = False


class nnUNetTrainerFinite(nnUNetTrainer):
    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions if self.label_manager.has_regions else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        # validation pipeline
        val_transforms = self.get_validation_transforms(
            None,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions if self.label_manager.has_regions else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)
        # Make the dataloaders finite (requirement by lightning)
        dl_tr.infinite = False
        dl_val.infinite = False
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes <= 1:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            # This is necessary for correctly sampling without replacement in each epoch
            dl_tr.number_of_threads_in_multithreaded = allowed_num_processes
            mt_gen_train = MultiThreadedAugmenterWithLength(
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            num_proc_val = max(1, allowed_num_processes // 2)
            dl_val.number_of_threads_in_multithreaded = num_proc_val
            mt_gen_val = MultiThreadedAugmenterWithLength(
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=num_proc_val,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val


class nnUNetTrainerNoDeepSupervisionFinite(nnUNetTrainerNoDeepSupervision, nnUNetTrainerFinite):
    pass


class nnUNetTrainerNoPatchesNoAugNoDeepsup(nnUNetTrainerNoDA, nnUNetTrainerNoDeepSupervision):
    def get_plain_dataloaders(self, num_processes: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = SimpleDataLoader(
            dataset_tr,
            batch_size=self.batch_size,
            label_manager=self.label_manager,
            num_threads_in_multithreaded=num_processes,
            shuffle=True,
            return_incomplete=True,
            infinite=False,
            patch_size=self.configuration_manager.patch_size,
        )
        dl_val = SimpleDataLoader(
            dataset_val,
            batch_size=self.batch_size,
            label_manager=self.label_manager,
            num_threads_in_multithreaded=num_processes,
            shuffle=False,
            return_incomplete=True,
            infinite=False,
            patch_size=self.configuration_manager.patch_size,
        )
        return dl_tr, dl_val

    def get_dataloaders(self):
        # Max: this is slightly adapted from the default nnunet trainer
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        allowed_num_processes = get_allowed_n_proc_DA()

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions if self.label_manager.has_regions else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        # validation pipeline
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions if self.label_manager.has_regions else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        dl_tr, dl_val = self.get_plain_dataloaders(allowed_num_processes)

        if allowed_num_processes <= 1:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            dl_tr.number_of_threads_in_multithreaded = allowed_num_processes
            mt_gen_train = MultiThreadedAugmenterWithLength(
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached_per_queue=1,
                seeds=None,
                pin_memory=self.device.type == "cuda",
            )
            num_proc_val = max(1, allowed_num_processes // 2)
            dl_val.number_of_threads_in_multithreaded = num_proc_val
            mt_gen_val = MultiThreadedAugmenterWithLength(
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=num_proc_val,
                seeds=None,
                pin_memory=self.device.type == "cuda",
            )
        return mt_gen_train, mt_gen_val


class FlexibleNumpyToTensor(NumpyToTensor):
    # Allows to cast to any torch dtype
    def cast(self, tensor):
        if isinstance(self.cast_to, torch.dtype):
            return tensor.to(self.cast_to)
        else:
            return super().cast(tensor)


def get_inference_transforms(regions=None):
    infer_transforms = []
    # the preprocessor inserts -1 labels, but for the purpose of evaluation they are actually background
    infer_transforms.append(RemoveLabelTransform(-1, 0, "target", "target"))

    if regions is not None:
        infer_transforms.append(ConvertSegmentationToRegionsTransform(regions, "target", "target"))

    infer_transforms.append(FlexibleNumpyToTensor(["data"], torch.float32))
    infer_transforms.append(FlexibleNumpyToTensor(["target"], torch.uint8))
    infer_transforms = Compose(infer_transforms)
    return infer_transforms


def reset_dataloader(dataloader):
    if isinstance(dataloader, SingleThreadedAugmenter):
        bg_dataloader = dataloader.data_loader
    elif isinstance(dataloader, MultiThreadedAugmenter):
        bg_dataloader = dataloader.generator
        # TODO not sure if I should also call restart here
    else:
        logger.warning(
            "Dataloader needs to be of type SingleThreadedAugmenter or MultiThreadedAugmenter. Cannot reset."
        )
    if not isinstance(bg_dataloader, DataLoader):
        logger.warning("Background dataloader needs to be of type DataLoader. Cannot reset.")
    else:
        bg_dataloader.reset()


# this is currently only used by quality regression and VAE data modules
# the actual nnunet dataloader uses nnunetv2 functionality.
# maybe switch to nnunetv2 for QR/VAE later, but it's not worth it right now.
class PreprocessImgSegAdapter(DataLoader):
    def __init__(
        self,
        data_dicts: list[dict[str, str]],
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        img_key: str = "data",
        seg_key: str = "target",
        num_threads_in_multithreaded: int = 1,
        verbose=False,
    ):
        super().__init__(
            data_dicts,
            1,
            num_threads_in_multithreaded,
            seed_for_shuffle=1,
            return_incomplete=True,
            shuffle=False,
            infinite=False,
            sampling_probabilities=None,
        )
        self.seg_key = seg_key
        self.img_key = img_key
        self.indices = list(range(len(data_dicts)))
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.preprocessor: DefaultPreprocessor = self.configuration_manager.preprocessor_class(
            verbose=verbose
        )
        self.label_manager = plans_manager.get_label_manager(dataset_json)

    def generate_train_batch(self):
        idx = self.get_indices()
        if len(idx) > 1:
            raise ValueError("Batch size must be 1")
        idx = idx[0]  # only one element in batch
        return self[idx]

    def load_case(self, case_id: str):
        for d in self._data:
            if d["keys"] == case_id:
                img_files = d[self.img_key]
                seg_file = d[self.seg_key]
                # process segmentation together with the images
                data, seg, data_properties = self.preprocessor.run_case(
                    img_files,
                    seg_file,
                    self.plans_manager,
                    self.configuration_manager,
                    self.dataset_json,
                )
                return data, seg, data_properties

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        img_files = self._data[idx][self.img_key]
        seg_file = self._data[idx][self.seg_key]
        case_id = self._data[idx]["keys"]
        domain = self._data[idx].get("domain_label", None)
        # process segmentation together with the images
        data, seg, data_properties = self.preprocessor.run_case(
            img_files,
            seg_file,
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )
        # both should have shape BCDWH (batch dim == 1; channel dim == 1 for seg)
        # expand the batch dim for both numpy arrays
        data = data[None]
        seg = seg[None]

        batch_dict = {
            "keys": [case_id],
            self.img_key: data,
            self.seg_key: seg,
            "properties": [data_properties],
        }
        if domain is not None:
            batch_dict["domain_label"] = [domain]
        return batch_dict
