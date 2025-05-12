import multiprocessing
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

# import torch.nn.functional as F
from loguru import logger
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
from pytorch_lightning.callbacks import Callback

from segmentation_failures.data.datamodules.monai_modules import MonaiBaseModule


class PredictionWriter(Callback):
    def __init__(
        self,
        output_dir: str,
        num_export_workers: int = 1,
        save_probabilities: bool = False,
        overwrite_existing=True,
    ):
        self.output_path = Path(output_dir)
        self.output_path.mkdir()
        self.save_probs = save_probabilities
        self.mp_pool = None
        self.overwrite = overwrite_existing
        if num_export_workers > 1:
            self.mp_pool = multiprocessing.get_context("spawn").Pool(num_export_workers)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if hasattr(trainer.datamodule, "nnunet_trainer"):
            pred = trainer.datamodule.post_inference_process(outputs["prediction"], batch)
            self._export_predictions_nnunet(pred, batch, trainer.datamodule.nnunet_trainer)
        elif isinstance(trainer.datamodule, MonaiBaseModule):
            self._export_predictions_simple(outputs, batch)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.state.fn == "validate":
            return self.on_test_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
        logger.debug(
            f"This callback is intended only for validation (trainer.validate), but the state.fn is {trainer.state.fn}."
        )

    def _export_predictions_nnunet(
        self,
        pred_tensor,
        batch,
        nnunet_trainer: nnUNetTrainer,
        case_id_suffix="",
        include_ids=None,
    ):
        # NOTE nnunet never exports one-hot masks
        # (it only knows region-based training, which is a special case of overlapping labels)
        # TODO nnunet has some precautions against too many exports at once. Maybe add later
        if include_ids is None:
            include_ids = batch["keys"]
        properties = batch["properties"]
        if len(batch["keys"]) == 1 and not isinstance(properties, list):
            properties = [properties]
        all_export_args = [
            (
                pred_tensor[idx],
                properties[idx],
                nnunet_trainer.configuration_manager,
                nnunet_trainer.plans_manager,
                nnunet_trainer.dataset_json,
                str(self.output_path / (case_id + case_id_suffix)),
                self.save_probs,
            )
            for idx, case_id in enumerate(batch["keys"])
            if case_id in include_ids
        ]
        if self.mp_pool is None:
            for args in all_export_args:
                export_prediction_from_logits(*args)
        else:
            self.mp_pool.starmap(
                export_prediction_from_logits,
                all_export_args,
            )

    def _export_predictions_simple(self, outputs, batch, case_id_suffix="", include_ids=None):
        if self.save_probs:
            raise NotImplementedError
        if include_ids is None:
            include_ids = batch["keys"]
        # this expects that predictions are given as logits
        predictions = outputs["prediction"]
        predictions = predictions.cpu()
        predictions = torch.argmax(predictions, dim=1).to(torch.uint8)
        # if self.save_probs:
        #     if has_regions:
        #         probabs = F.sigmoid(predictions, dim=1)
        #     else:
        #         probabs = F.softmax(predictions, dim=1)
        affine = np.eye(4)
        if hasattr(batch["data"], "meta"):
            affine = batch["data"].meta["affine"][0].cpu().numpy()
            # assume all have the same affine matrix in batch
        for pred, case_id in zip(predictions, batch["keys"]):
            if case_id not in include_ids:
                continue
            # np.savez_compressed(self.output_path / case_id, pred.cpu().numpy())
            nib.save(
                nib.Nifti1Image(pred.cpu().numpy(), affine=affine),
                self.output_path / (case_id + f"{case_id_suffix}.nii.gz"),
            )

    def teardown(self, trainer, pl_module, stage: str) -> None:
        if self.mp_pool is not None:
            self.mp_pool.close()


class MultiPredictionWriter(PredictionWriter):
    # for saving multiple predictions per case (e.g. in an ensemble)
    # This could be combined with PredictionWriter, but I don't want to touch the latter for now
    def __init__(self, pred_key: str, **kwargs):
        super().__init__(**kwargs)
        self.pred_key = pred_key

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.pred_key not in outputs:
            # I'm not escalating this here assuming that this callback is not essential
            logger.warning(
                f"Could not find key {self.pred_key} in batch output. Skipping export of ensemble predictions."
            )
            return
        if hasattr(trainer.datamodule, "nnunet_trainer"):
            pred = trainer.datamodule.post_inference_process(
                outputs[self.pred_key], batch, batch_dim=1
            )
            self._export_predictions_nnunet(pred, batch, trainer.datamodule.nnunet_trainer)
        elif isinstance(trainer.datamodule, MonaiBaseModule):
            self._export_predictions_simple(outputs, batch)

    def _export_predictions_nnunet(
        self,
        predictions,
        batch,
        nnunet_trainer: nnUNetTrainer,
        case_id_suffix="",
        include_ids=None,
    ):
        # NOTE nnunet never exports one-hot masks
        # (it only knows region-based training, which is a special case of overlapping labels)
        # TODO nnunet has some precautions against too many exports at once. Maybe add later
        if include_ids is None:
            include_ids = batch["keys"]
        properties = batch["properties"]
        if len(batch["keys"]) == 1 and not isinstance(properties, list):
            properties = [properties]
        # prediction shape KBCHW[D], K=#models in ensemble
        if len(batch["keys"]) != predictions.shape[1]:
            raise ValueError(
                f"Expected that predictions have shape (num_preds, num_cases, ...), but got {predictions.shape}."
            )
        num_preds_per_case = predictions.shape[0]
        assert num_preds_per_case < 100
        all_export_args = []
        for batch_idx, case_id in enumerate(batch["keys"]):
            if case_id not in include_ids:
                continue
            for pred_idx in range(num_preds_per_case):
                all_export_args.append(
                    (
                        predictions[pred_idx, batch_idx],
                        properties[batch_idx],
                        nnunet_trainer.configuration_manager,
                        nnunet_trainer.plans_manager,
                        nnunet_trainer.dataset_json,
                        str(self.output_path / f"{case_id}{case_id_suffix}_{pred_idx:02d}"),
                        self.save_probs,
                    )
                )
        if self.mp_pool is None:
            for args in all_export_args:
                export_prediction_from_logits(*args)
        else:
            self.mp_pool.starmap(
                export_prediction_from_logits,
                all_export_args,
            )

    def _export_predictions_simple(self, outputs, batch, case_id_suffix="", include_ids=None):
        if self.save_probs:
            raise NotImplementedError
        if include_ids is None:
            include_ids = batch["keys"]
        # this expects that predictions are given as logits
        predictions = outputs[self.pred_key]
        # prediction shape KBCHW[D], K=#models in ensemble
        assert len(predictions) < 100
        predictions = predictions.detach()
        # NOTE multi-label is not supported
        predictions = torch.argmax(predictions, dim=2).to(torch.uint8)
        affine = np.eye(4)
        if hasattr(batch["data"], "meta"):
            affine = batch["data"].meta["affine"][0].cpu().numpy()
            # assume all have the same affine matrix in batch
        for batch_idx, case_id in enumerate(batch["keys"]):
            if case_id not in include_ids:
                continue
            for pred_idx in range(predictions.shape[0]):
                nib.save(
                    nib.Nifti1Image(predictions[pred_idx, batch_idx].cpu().numpy(), affine=affine),
                    self.output_path / f"{case_id}{case_id_suffix}_{pred_idx:02d}.nii.gz",
                )

    def teardown(self, trainer, pl_module, stage: str) -> None:
        if self.mp_pool is not None:
            self.mp_pool.close()


class PredictionWriterWithBalancing(PredictionWriter):
    """
    This class is used for generating quality regression masks of diverse Dice scores.
    """

    def __init__(
        self,
        output_dir: str,
        num_fg_classes: int,
        num_export_workers: int = 1,
        num_bins: int = 20,
        max_num_per_bin: int = 2,
        randomize_bins=True,
    ):
        """
        Initializes a PredictionWriterVariant object.

        Args:
            output_dir (str): The directory where the output files will be saved.
            num_fg_classes (int): The number of foreground classes.
            num_export_workers (int, optional): The number of workers to use for exporting. Defaults to 1.
            num_bins (int, optional): The number of bins to use for dice score histogram. Defaults to 10.
            max_num_per_bin (int, optional): The maximum number of predictions to save per bin and case id. Defaults to 1.
                Therefore, the maximum total number of saved predictions per case id is num_bins * max_num_per_bin.
        """
        super().__init__(
            output_dir, num_export_workers, save_probabilities=False, overwrite_existing=False
        )
        self.dice_counts = defaultdict(lambda: np.zeros(num_bins, dtype=np.int8))
        assert max_num_per_bin < 2**8
        self.dice_bin_edges = np.linspace(0, 1, num_bins + 1)
        self.max_bin_count = max_num_per_bin
        self.num_classes = num_fg_classes
        self.dice_values_buffer = []
        self.randomize_bins = randomize_bins

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError("This callback is only used for validation.")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.sanity_checking:
            return
        if not self.overwrite:
            case_id_suffix = f"_{trainer.current_epoch:04d}"
        if "dice" not in outputs:
            raise ValueError("Lightning module did not output dice scores.")
        dice_scores = outputs["dice"].detach().cpu().numpy()  # shape BC
        # Compute the mean dice per batch (only foreground classes)
        mean_dice = dice_scores[:, -self.num_classes :].mean(axis=1)
        selected_cases = []
        num_bins = len(self.dice_bin_edges) - 1
        for i, case_id in enumerate(batch["keys"]):
            curr_bin_edges = self.dice_bin_edges
            if self.randomize_bins:
                curr_bin_edges = self.dice_bin_edges + (np.random.uniform() - 0.5) / num_bins
            # Get the count of the bin this batch falls into
            bin_idx = np.clip(
                np.digitize(mean_dice[i], curr_bin_edges) - 1, a_min=0, a_max=num_bins - 1
            )
            if self.dice_counts[case_id][bin_idx] >= self.max_bin_count:
                logger.debug(
                    f"Not saving {case_id} with Dice {mean_dice[i]} "
                    f"because there are enou1gh batches with this dice score."
                )
            else:
                self.dice_counts[case_id][bin_idx] += 1
                self.dice_values_buffer.append(
                    {"prediction": case_id + case_id_suffix, "mean_dice": mean_dice[i]}
                )
                selected_cases.append(case_id)
        if hasattr(trainer.datamodule, "nnunet_trainer"):
            self._export_predictions_nnunet(
                outputs, batch, trainer.datamodule.nnunet_trainer, case_id_suffix, selected_cases
            )
        elif isinstance(trainer.datamodule, MonaiBaseModule):
            self._export_predictions_simple(outputs, batch, case_id_suffix, selected_cases)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # save the dice values
        dice_values_file = self.output_path / "dice_values.csv"
        complete_df = pd.DataFrame(self.dice_values_buffer)
        if dice_values_file.exists():
            complete_df = pd.concat(
                [pd.read_csv(dice_values_file), complete_df], axis=0, ignore_index=True
            )
        # concatenate the previous and current dice values
        if len(complete_df) > 0:
            complete_df.to_csv(dice_values_file, index=False)
        self.dice_values_buffer = []
