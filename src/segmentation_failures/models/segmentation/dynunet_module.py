import numpy as np
import pytorch_lightning as pl
import torch
import torchio as tio
from loguru import logger
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

from segmentation_failures.utils.data import get_padding


class DynUnetModule(pl.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        num_classes: int,
        patch_size: tuple[int] | list[int],
        batch_dice: bool = False,
        sw_batch_size=1,
        sw_overlap=0.5,
        lr=1e-3,
        weight_decay=3e-5,
        overlapping_classes: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="backbone")
        self.model = backbone
        dice_loss_kwargs = {
            "batch_dice": batch_dice,
            "do_bg": True,
            "smooth": 1e-5,
            "ddp": False,
        }
        self.loss_deepsup = None  # initialized in on_train_start
        self.loss_function = DC_and_CE_loss(
            ce_kwargs={},
            soft_dice_kwargs=dice_loss_kwargs,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        if overlapping_classes:
            self.loss_function = DC_and_BCE_loss(
                bce_kwargs={},
                soft_dice_kwargs=dice_loss_kwargs,
                use_ignore_label=False,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        # Cases/classes with missing GT are set to nan
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean_batch", get_not_nans=False
        )
        if overlapping_classes:
            self.post_pred = Compose(
                [EnsureType(), AsDiscrete(threshold=0)]
            )  # log p > 0 => p > 0.5
            self.post_label = EnsureType()  # already in correct format
        else:
            self.post_pred = Compose(
                [EnsureType(), AsDiscrete(argmax=True, to_onehot=num_classes, dim=1)]
            )
            self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes, dim=1)])

    def on_train_start(self) -> None:
        if self.trainer.datamodule.deep_supervision:
            deep_supervision_scales = (
                self.trainer.datamodule.nnunet_trainer._get_deep_supervision_scales()
            )
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            self.loss_deepsup = DeepSupervisionWrapper(self.loss_function, weights)

    def sliding_window_inference(self, x: torch.Tensor):
        # only batch size 1 supported
        try:
            pred = torch.zeros(
                (x.shape[0], self.hparams.num_classes, *x.shape[2:]), device=self.device
            )
            for i in range(x.shape[0]):
                pred[i] = self._sliding_window_inference(x[i], out_device=self.device)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("CUDA out of memory. Falling back to CPU.")
                # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                pred = torch.zeros(
                    (x.shape[0], self.hparams.num_classes, *x.shape[2:]), device="cpu"
                )
                for i in range(x.shape[0]):
                    pred[i] = self._sliding_window_inference(x[i], out_device="cpu")
            else:
                raise e
        return pred

    # def _sliding_window_inference_not(self, x: torch.Tensor):
    #     # this is just for testing purposes (what happens if no sliding window inference is used?)
    #     test_subject = tio.Subject(image=tio.ScalarImage(tensor=x))
    #     # pad image to patch size if necessary
    #     pad_sizes = get_padding(test_subject.spatial_shape, (320, 320, 32))
    #     if any([ps > 0 for ps in pad_sizes]):
    #         test_subject = tio.transforms.Pad(padding=pad_sizes)(test_subject)
    #     pred = self.model(test_subject.image.data.to(x.device).unsqueeze(0))
    #     prediction = tio.transforms.Crop(cropping=pad_sizes)(pred.squeeze(0))
    #     return prediction.to(device=x.device)

    def _sliding_window_inference(self, x: torch.Tensor, out_device=None):
        if out_device is None:
            out_device = self.device
        # assumes shape (C, H, W, D)
        patch_size = self.hparams.patch_size
        if patch_size == list(x.shape[-3:]):
            # no sliding window inference necessary
            return self.model(x.unsqueeze(0).to(self.device)).squeeze(0).to(out_device)
        sw_overlap = self.hparams.sw_overlap
        if isinstance(sw_overlap, float):
            sw_overlap = tuple(int(self.hparams.sw_overlap * ps) for ps in patch_size)
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
            patch_aggregator.add_batch(pred, locations)
        del patches_batch
        # aggregate
        output = patch_aggregator.get_output_tensor()
        assert output.ndim == 4  # CHWD. Assumption of single test subject (see top of forward)
        # back to original dims
        prediction = tio.transforms.Crop(cropping=pad_sizes)(output)
        return prediction.to(out_device)

    def forward(self, x):
        # this is used for inference!
        # NOTE this might return logits on GPU or CPU. The caller has to check.
        spatial_size = x.shape[2:]
        expected_size = getattr(self.model, "spatial_dims", spatial_size)
        if hasattr(expected_size, "__len__"):
            expected_size = len(expected_size)
        if expected_size != len(spatial_size):
            raise ValueError(
                f"Got an input image with {x.dim()}, but expected "
                f"{2 + expected_size} (batch + channel + spatial dims)"
            )
        if len(spatial_size) == 2:
            # 2D case: assume that no sliding window is necessary
            if any([d1 < d2 for d1, d2 in zip(self.hparams.patch_size, spatial_size)]):
                raise NotImplementedError("Sliding window inference not implemented for 2D.")
            # no sliding window necessary
            return self.model(x)
        return self.sliding_window_inference(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=0.99,
            weight_decay=self.hparams.weight_decay,
            nesterov=True,
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: (1 - epoch / self.trainer.max_epochs) ** 0.9,
            ),
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def training_step(self, batch, batch_idx):
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        targets = batch["target"]
        preds = self.model(batch["data"])
        if isinstance(targets, list):
            if not isinstance(preds, list) or len(targets) != len(preds):
                raise ValueError(
                    f"Deep supervision requires list of predictions and targets to have same shape. "
                    f"Got {type(preds)} (len {len(preds)}) and {type(targets)} (len {len(targets)})"
                )
            # deep supervision
            loss = self.loss_deepsup(preds, targets)
            preds = preds[0]  # no need for other scales anymore
        else:
            loss = self.loss_function(preds, targets)
        self.log("train_loss", loss, on_epoch=True, batch_size=len(batch["data"]))
        return {"loss": loss, "prediction": preds.detach()}

    def validation_step(self, batch, batch_idx):
        # Validation is simplified compared to the original dynunet pipeline:
        # - I don't allow TTA
        # - I don't resample/pad to original image spacing/size (just sliding window)
        images, labels = batch["data"], batch["target"]
        # NOTE during training, these will be patches; during validation full images.
        # NOTE: deep supervision should always be disabled here (on the model side)
        if self.trainer.state.fn == "fit":
            # I added this because I got out-of-memory issues when computing the loss
            # with the full image after sliding window inference (nnunet dataloader).
            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)
            self.log("val_loss_epoch", loss, batch_size=len(labels))
            dice_scores = self.compute_dice(outputs, labels)
        elif self.trainer.state.fn == "validate":
            outputs = self(images)  # inference routine
            loss = None
            dice_scores = self.compute_dice(outputs, labels, use_cpu=True)
        else:
            raise NotImplementedError(f"Validation step not trainer state {self.trainer.state.fn}")
        return {"loss": loss, "prediction": outputs, "dice": dice_scores}

    def on_validation_epoch_end(self):
        if len(self.dice_metric) == 0:
            # no dice computed
            logger.warning("Empty validation metric (dice) buffer. Skipping dice logging.")
            return
        mean_val_dice_per_class = self.dice_metric.aggregate().cpu()
        self.dice_metric.reset()
        logger.info(f"current mean dice (per class): {mean_val_dice_per_class}")
        self.log_dict(
            {f"val_dice_class{i}": dice.item() for i, dice in enumerate(mean_val_dice_per_class)}
        )

    def test_step(self, batch, batch_idx):
        images = batch["data"]
        labels = batch.get("target", None)
        outputs = self(images)
        result = {"prediction": outputs}
        if labels is not None:
            dice_scores = self.compute_dice(outputs, labels)
            result["dice"] = dice_scores
        return result

    def on_test_epoch_end(self):
        if self.dice_metric.get_buffer() is None:
            # no dice computed
            logger.warning("Empty test metric (dice) buffer. Skipping dice logging.")
            return
        mean_dice_per_class = self.dice_metric.aggregate().cpu()
        self.dice_metric.reset()
        self.log_dict(
            {f"test_dice_class{i}": dice.item() for i, dice in enumerate(mean_dice_per_class)}
        )

    @torch.no_grad()
    def compute_dice(self, prediction, target, use_cpu=False):
        prediction = prediction.detach()
        target = target.detach()
        if use_cpu:
            prediction = prediction.cpu()
            target = target.cpu()
        return self.dice_metric(y_pred=self.post_pred(prediction), y=self.post_label(target))
