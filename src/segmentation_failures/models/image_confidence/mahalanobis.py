# Adapted from https://github.com/MECLabTUDA/Lifelong-nnUNet/blob/dev-ood_detection/nnunet_ext/calibration/mahalanobis
# ------------------------------------------------------------------------------
# Multivariate density estimators.
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from loguru import logger
from pytorch_lightning import LightningModule
from sklearn.covariance import EmpiricalCovariance

from segmentation_failures.utils.data import get_padding
from segmentation_failures.utils.feature_extraction import ActivationSeeker


class SingleGaussianOODDetector(LightningModule):
    def __init__(
        self,
        segmentation_net: torch.nn.Module,
        feature_path: str | list[str],
        sw_patch_size: list[int] | None = None,
        sw_batch_size: int = 1,
        sw_overlap: float = 0.5,
        sw_training: bool = False,
        max_feature_size=10000,
        store_precision=True,
        assume_centered=False,
    ):
        """Initialize the Gaussian OOD detector from Gonzalez et al. (2022)

        Args:
            model (torch.nn.Module, optional): Segmentation model. Defaults to None.
            feature_path (str, optional): Path to the module in the segmentation model whose features are used
                for fitting the Gaussian.
            sw_patch_size: Can be None if no sliding window should be used (e.g. 2D)
            sw_batch_size (int): number of patches per batch for sliding window inference
            sw_overlap (float): fractional overlap between sw patches.
            max_feature_size (int, optional): Maximum dimension of features used for fitting the Gaussian. Defaults to 10000.
            store_precision (bool, optional): Passed to sklearn.covariance.EmpiricalCovariance. Defaults to True.
            assume_centered (bool, optional): Passed to sklearn.covariance.EmpiricalCovariance. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["segmentation_net"])
        self.max_feature_size = max_feature_size
        self.feature_extractor = ActivationSeeker(to_cpu=True)
        if isinstance(feature_path, str):
            feature_path = [feature_path]
        self.feature_paths = sorted(feature_path)
        self.gaussian_estimators = {
            k: EmpiricalCovariance(
                store_precision=store_precision, assume_centered=assume_centered
            )
            for k in self.feature_paths
        }
        self._model = segmentation_net.model  # NOTE this is the nn.Module, not lightning module
        self.feature_extractor.attach_hooks(self._model, {k: k for k in self.feature_paths})

        # TODO above assumes that the model attribute exists.
        # I do this because I don't want to use custom inference procedure by the segmentation module,
        # but in the future it would be better to make the inference procedure configurable (?)
        self._dummy_parameter = torch.nn.Parameter(data=torch.zeros(2), requires_grad=True)
        if isinstance(sw_overlap, float) and sw_patch_size is not None:
            sw_overlap = tuple(int(sw_overlap * ps) for ps in sw_patch_size)
            sw_overlap = [i + i % 2 for i in sw_overlap]  # make even. WHY does tio need this?
        self.sw_patch_size = sw_patch_size
        self.sw_overlap = sw_overlap
        self.all_features = {k: [] for k in self.feature_paths}

    @property
    def model(self):
        if self._model is None:
            raise AttributeError("Model is not initialized. Please call `set_model` first!")
        return self._model

    def _post_process_features(self, x: torch.Tensor):
        # Expects tensor with shape BCXY[Z]
        # Apply average pooling to reduce dimensionality
        size_without_batch_dim = torch.numel(x[0, :])
        while size_without_batch_dim > self.max_feature_size:
            if len(x.shape) == 4:  # 2D
                kernel_size = [min(5, s) for s in x.shape[2:]]
                x = F.avg_pool2d(x, kernel_size, stride=(3, 3))
            elif len(x.shape) == 5:  # 3D
                kernel_size = [min(2, s) for s in x.shape[2:]]
                x = F.avg_pool3d(x, kernel_size, stride=(2, 2, 2))
            else:
                raise ValueError(
                    f"Got shape {x.shape} with length {len(x.shape)}, but expected it to have length 4 or 5."
                )
            size_without_batch_dim = torch.numel(x[0, :])

        return x.flatten(start_dim=1)

    def _get_features(self):
        feature_dict = self.feature_extractor.get_data_activations()
        for k in feature_dict:
            feature_dict[k] = self._post_process_features(feature_dict[k])
        return feature_dict

    def _sliding_window_inference(self, x: torch.Tensor, out_device=None):
        if out_device is None:
            out_device = self.device
        # assumes shape (C, H, W, D)
        patch_size = self.sw_patch_size
        sw_overlap = self.sw_overlap
        if isinstance(sw_overlap, float):
            sw_overlap = tuple(int(self.sw_overlap * ps) for ps in patch_size)
            sw_overlap = [i + i % 2 for i in sw_overlap]  # make even. WHY does tio need this?
        # subject needs 4D tensor (CHWD)
        test_subject = tio.Subject(image=tio.ScalarImage(tensor=x.cpu()))  # tio needs cpu tensors
        # pad image to patch size if necessary
        pad_sizes = get_padding(test_subject.spatial_shape, patch_size)
        if any([ps > 0 for ps in pad_sizes]):
            test_subject = tio.transforms.Pad(padding=pad_sizes)(test_subject)

        # sliding window sampling
        sw_sampler = tio.inference.GridSampler(
            subject=test_subject,
            patch_size=patch_size,
            patch_overlap=sw_overlap,
        )
        patch_loader = torch.utils.data.DataLoader(
            sw_sampler, batch_size=self.hparams.sw_batch_size, shuffle=False, pin_memory=True
        )
        patch_aggregator = tio.inference.GridAggregator(sw_sampler, overlap_mode="hann")
        # inference on sliding window patches
        for patches_batch in patch_loader:
            # move data back to original device (usually GPU) for prediction
            img_data = patches_batch["image"][tio.DATA].to(self.device).detach()
            pred = self.model(img_data).to(out_device)
            locations = patches_batch[tio.LOCATION]
            feature_dict = self._get_features()
            pseudo_pixel_confids = []
            for k in sorted(feature_dict):
                patch_confid = -torch.as_tensor(
                    self.gaussian_estimators[k].mahalanobis(feature_dict[k].cpu().numpy()),
                    device=out_device,
                )  # Shape B
                # stack confidence and prediction in channel dimension
                patch_confid = patch_confid.reshape([len(patch_confid)] + [1] * (pred.ndim - 1))
                pseudo_pixel_confids.append(patch_confid.expand((-1, 1, *pred.shape[2:])))
            patch_aggregator.add_batch(
                torch.concat([pred] + pseudo_pixel_confids, dim=1), locations
            )
        # aggregate
        output = patch_aggregator.get_output_tensor().to(out_device)
        assert output.ndim == 4  # CHWD. Assumption of single test subject (see top of forward)
        prediction, pixel_confid = torch.split(
            output, [len(output) - len(self.feature_paths), len(self.feature_paths)], dim=0
        )
        # back to original dims
        prediction = tio.transforms.Crop(cropping=pad_sizes)(prediction)
        pixel_confid = tio.transforms.Crop(cropping=pad_sizes)(pixel_confid)
        # NOTE I could in principle also allow other aggregations but this is what Gonzalez et al. did
        confidence = {}
        for idx, k in enumerate(self.feature_paths):
            confidence[k] = pixel_confid[idx].mean()
        return prediction, confidence

    def _sliding_window_feature_extraction(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # assumes shape (C, H, W, D)
        patch_size = self.sw_patch_size
        sw_overlap = self.sw_overlap
        if isinstance(sw_overlap, float):
            sw_overlap = tuple(int(self.sw_overlap * ps) for ps in patch_size)
            sw_overlap = [i + i % 2 for i in sw_overlap]  # make even. WHY does tio need this?
        # subject needs 4D tensor (CHWD)
        test_subject = tio.Subject(image=tio.ScalarImage(tensor=x.cpu()))  # tio needs cpu tensors
        # pad image to patch size if necessary
        pad_sizes = get_padding(test_subject.spatial_shape, patch_size)
        if any([ps > 0 for ps in pad_sizes]):
            test_subject = tio.transforms.Pad(padding=pad_sizes)(test_subject)

        # sliding window sampling
        sw_sampler = tio.inference.GridSampler(
            subject=test_subject,
            patch_size=patch_size,
            patch_overlap=sw_overlap,
        )
        patch_loader = torch.utils.data.DataLoader(
            sw_sampler, batch_size=self.hparams.sw_batch_size, shuffle=False, pin_memory=True
        )
        # inference on sliding window patches
        all_features = {k: [] for k in self.feature_paths}
        for patches_batch in patch_loader:
            img_data = patches_batch["image"][tio.DATA]
            self.model(img_data.to(self.device))  # move back to (usually) GPU
            curr_feature_dict = self._get_features()
            for k in all_features:
                all_features[k].append(curr_feature_dict[k])
        feature_dict = {k: torch.cat(all_features[k], dim=0) for k in all_features}
        return feature_dict

    def forward(self, x: torch.Tensor):
        """Inference forward pass

        Args:
            x (torch.Tensor): Shape BCHW[D]

        Returns:
            tuple of prediction logits and pixel confidence
        """
        if x.ndim == 4 or self.sw_patch_size is None:
            # 2D case
            prediction = self.model(x)
            feature_dict = self._get_features()
            confid = {}
            for k, features in feature_dict.items():
                confid[k] = -torch.as_tensor(
                    self.gaussian_estimators[k].mahalanobis(features.cpu().numpy())
                )  # Shape B
            return prediction, confid
        # 3D case; only batch size 1 supported
        return self.forward_3d(x, out_device="cpu")

    def forward_3d(self, x: torch.Tensor, out_device):
        for i in range(x.shape[0]):
            curr_pred, curr_confid = self._sliding_window_inference(x[i], out_device=out_device)
            if i == 0:
                pred = torch.zeros((x.shape[0], *curr_pred.shape), device=out_device)
                confid = {
                    k: torch.zeros((x.shape[0], *curr_confid[k].shape), device=out_device)
                    for k in curr_confid
                }
            pred[i] = curr_pred
            for k in curr_confid:
                confid[k][i] = curr_confid[k]
        return pred, confid

    # this model does not require gradients. Only one epoch is needed
    def training_step(self, batch, batch_idx):
        # The segmentation model should be fixed (also don't update BN statistics)
        self.model.eval()
        with torch.no_grad():
            if self.hparams.sw_training and batch["data"].ndim > 4:
                assert batch["data"].shape[0] == 1
                feature_dict = self._sliding_window_feature_extraction(batch["data"].squeeze(0))
            else:
                self.model(batch["data"])  # hook extracts features
                feature_dict = self._get_features()
        for k in self.feature_paths:
            self.all_features[k].append(feature_dict[k].cpu().numpy())
        # for dummy optimizer
        loss = self._dummy_parameter.sum().requires_grad_()
        return {"loss": loss, "features": feature_dict}

    def on_train_epoch_end(self):
        if self.hparams.sw_training and self.current_epoch > 1:
            logger.warning(
                "This model only needs one epoch to train. Duplicate features are being produced!"
            )
        if self.current_epoch == self.trainer.max_epochs - 1:
            logger.info("Fitting multivariate Gaussian to extracted features...")
            for k, all_features in self.all_features.items():
                feature_array = np.concatenate(all_features, axis=0)
                self.gaussian_estimators[k].fit(feature_array)

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        prediction, confidence = self(batch["data"])  # Shapes BCHW[D] and B1HW[D]
        return {
            "prediction": prediction,
            "confidence": confidence,
            "confidence_pixel": None,
        }

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["gaussian_estimator"] = {
            k: self.gaussian_estimators[k].__dict__ for k in self.feature_paths
        }

    def on_load_checkpoint(self, checkpoint) -> None:
        gaussian_state_dict = checkpoint["gaussian_estimator"]
        for k in self.feature_paths:
            self.gaussian_estimators[k].__dict__ = gaussian_state_dict[k]

    # to make lightning happy...
    def configure_optimizers(self):
        return torch.optim.SGD([self._dummy_parameter], lr=0)
