from abc import abstractmethod
from collections import defaultdict
from copy import copy

import sklearn.metrics as skmetrics
import torch
from loguru import logger
from pytorch_lightning import LightningModule
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from segmentation_failures.models.confidence_aggregation.base import (
    AbstractAggregator,
    BackgroundAggregator,
    BoundaryAggregator,
    ConnectedComponentsAggregator,
    ForegroundAggregator,
    ForegroundSizeAggregator,
)


def get_regression_model(model_name):
    # would be nice to make this more configurable
    if model_name == "regression_forest":
        return RandomForestRegressor()
    else:
        raise ValueError(f"Unknown regression model {model_name}")


class AbstractHeuristicAggregationModule(LightningModule):
    def __init__(
        self,
        regression_model: str,
        dataset_id: int,
        confid_name: str,
        target_metrics: list[str],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.quality_estimator = make_pipeline(
            StandardScaler(),
            get_regression_model(regression_model),
        )
        self._dummy_parameter = torch.nn.Parameter(data=torch.zeros(2), requires_grad=True)
        self._training_buffer = []
        self._validation_buffer = []
        self.confid_name = confid_name
        self.target_metrics = target_metrics  # just for output naming
        # determine which class-wise metrics need to be averaged to get mean_metric
        self.metrics_to_average = defaultdict(list)
        for idx, name in enumerate(target_metrics):
            if name.split("_")[-1].isdigit():
                # class-wise metric format: {metric_name}_{class_idx}
                metric_name = "_".join(name.split("_")[:-1])
                self.metrics_to_average[metric_name].append(idx)

    @abstractmethod
    def extract_features(self, prediction, pxl_confid):
        """Abstract interface for extracting features from the (prediction, confidence) maps

        Args:
            prediction (torch.Tensor): label map. Expected shape BHW[D]
            pxl_confid (torch.Tensor): Expected shape BHW[D]
        """
        pass

    def training_step(self, batch, batch_idx):
        """This does not really train but just computes features and targets.
        Actual training is in training_epoch_end.
        Batch should have keys "pred", "confid", "metric_target"
        """
        # shapes: pred B1HW[D], confid B1HW[D]
        assert batch["pred"].shape[1] == 1, batch["pred"].shape
        assert batch["confid"].shape[1] == 1, batch["confid"].shape
        features = self.extract_features(
            batch["pred"].squeeze(1), batch["confid"].squeeze(1)
        )  # shape BF
        # for dummy optimizer
        loss = self._dummy_parameter.sum().requires_grad_()
        self._training_buffer.append({"features": features, "targets": batch["metric_target"]})
        return {
            "loss": loss,
            "quality_true": batch["metric_target"],
        }

    def on_train_epoch_end(self):
        if self.current_epoch > 1:
            print(
                "WARNING: this model only needs one epoch to train, so you're wasting time right now."
            )
        feature_array = (
            torch.cat([out["features"] for out in self._training_buffer], dim=0).cpu().numpy()
        )
        target_array = (
            torch.cat([out["targets"] for out in self._training_buffer], dim=0)
            .cpu()
            .numpy()
            .squeeze()
        )
        self.quality_estimator.fit(feature_array, target_array)
        y_pred = self.quality_estimator.predict(feature_array)
        mse = skmetrics.mean_squared_error(target_array, y_pred)
        r2_coeff = skmetrics.r2_score(target_array, y_pred)
        self.log("train_MSE", mse)
        self.log("train_R2", r2_coeff)
        if len(self._validation_buffer) > 0:
            self.hacky_validation_epoch_end()
        else:
            logger.debug("Skipping validation as there is no data for it.")
        self._training_buffer = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        assert batch["pred"].shape[1] == 1, batch["pred"].shape
        assert batch["confid"].shape[1] == 1, batch["confid"].shape
        features = self.extract_features(batch["pred"].squeeze(1), batch["confid"].squeeze(1))
        targets = batch["metric_target"]
        assert targets.shape[1] == len(self.target_metrics)
        self._validation_buffer.append({"features": features, "targets": targets})

    def test_step(self, batch, batch_idx):
        assert batch["pred"].shape[1] == 1, batch["pred"].shape
        assert batch["confid"].shape[1] == 1, batch["confid"].shape
        confidence_features = self.extract_features(
            batch["pred"].squeeze(1), batch["confid"].squeeze(1)
        )
        estimated_quality = self.quality_estimator.predict(confidence_features.cpu().numpy())  # BN
        confid_names = copy(self.target_metrics)
        assert len(confid_names) == estimated_quality.shape[1]
        higher_better = (
            2 * torch.tensor(self.trainer.datamodule.metric_higher_better).unsqueeze(0) - 1
        )
        confidence = higher_better * torch.tensor(estimated_quality)
        # add mean_metric to the output
        for metric_name, avg_idxs in self.metrics_to_average.items():
            confid_names.append(f"mean_{metric_name}")
            mean_confid = confidence[:, avg_idxs].mean(dim=1)
            confidence = torch.cat([confidence, mean_confid.unsqueeze(dim=1)], dim=1)
        return {
            "prediction": batch["pred"],
            "confidence": {k: confidence[:, i] for i, k in enumerate(confid_names)},
        }

    # HACK validation_epoch_end is called after training_epoch_end in the default loop.
    # Using this dirty solution, training_epoch_end is called before hacky_validation_epoch_end.
    def hacky_validation_epoch_end(self):
        # Evaluate quality estimator model
        feature_array = (
            torch.cat([out["features"] for out in self._validation_buffer], dim=0).cpu().numpy()
        )
        target_array = (
            torch.cat([out["targets"] for out in self._validation_buffer], dim=0).cpu().numpy()
        ).squeeze()
        y_pred = self.quality_estimator.predict(feature_array)
        mse = skmetrics.mean_squared_error(target_array, y_pred)
        r2_coeff = skmetrics.r2_score(target_array, y_pred)
        self.log("val_MSE", mse)
        self.log("val_R2", r2_coeff)
        self.log(
            "val_loss_epoch", mse, on_epoch=True, on_step=False
        )  # just to have something here...
        # reset
        self._validation_buffer = []

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["quality_estimator"] = self.quality_estimator.__dict__

    def on_load_checkpoint(self, checkpoint) -> None:
        estimator = checkpoint["quality_estimator"]
        self.quality_estimator.__dict__ = estimator

    # to make lightning happy...
    def configure_optimizers(self):
        return torch.optim.SGD([self._dummy_parameter], lr=0)


class HeuristicAggregationModule(AbstractHeuristicAggregationModule):
    def __init__(
        self,
        heuristic_list: list[AbstractAggregator] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if heuristic_list is None:
            heuristic_list = [
                BoundaryAggregator(boundary_width=4),
                ForegroundAggregator(boundary_width=4),
                BackgroundAggregator(boundary_width=4),
                ForegroundSizeAggregator(fractional_size=True),
                ConnectedComponentsAggregator(),
            ]
        self.aggregator_list = heuristic_list

    def extract_features(self, prediction, pxl_confid):
        # prediction and confid shape BHW[D]
        # As I only use AbstractAggregators, I can use the label maps directly
        confidence_features = []
        for agg in self.aggregator_list:
            confidence_features.append(agg(prediction, pxl_confid).to(self.device))
        return torch.stack(confidence_features, dim=1)  # Batch x num_features
