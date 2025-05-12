import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from segmentation_failures.data.datamodules.nnunet_module import NNunetDataModule


class DummyDataset(Dataset):
    def __init__(self, data, target_data, spacing=(1, 1, 1)):
        self.data = data
        self.target_data = target_data
        self.spacing = spacing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_size = self.data[idx].shape[1:]
        return {
            "data": self.data[idx],
            "target": self.target_data[idx],
            "keys": f"sample_{idx}",
            "properties": [
                {
                    "spacing": np.array(self.spacing),
                    "shape_after_cropping_and_before_resampling": img_size,
                    "shape_before_cropping": img_size,
                    "bbox_used_for_cropping": [[i, img_size[i]] for i in range(len(img_size))],
                }
            ],
        }


class DummyNNunetDataModule(NNunetDataModule):
    # NOTE: the batch collation by pytorch converts properties to tensors, which causes problems with some callbacks;
    # currently need to disable them when using this module
    def __init__(
        self,
        dummy_num_samples: int,
        dummy_num_channels: int,
        dummy_img_size: list[int],
        dummy_batch_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dummy_batch_size = dummy_batch_size
        xs = torch.randn(dummy_num_samples, dummy_num_channels, *dummy_img_size)
        if self.nnunet_trainer.label_manager.has_regions:
            num_classes = len(self.nnunet_trainer.label_manager.foreground_regions)
            ys = torch.randint(0, 2, size=(dummy_num_samples, num_classes, *dummy_img_size))
        else:
            num_classes = len(self.nnunet_trainer.label_manager.foreground_labels)
            ys = torch.randint(0, num_classes, size=(dummy_num_samples, 1, *dummy_img_size))
        self.dummy_dataset = DummyDataset(xs, ys, spacing=self.preprocess_info["spacing"])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.dummy_dataset, batch_size=self.dummy_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dummy_dataset, batch_size=self.dummy_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dummy_dataset, batch_size=self.dummy_batch_size)
