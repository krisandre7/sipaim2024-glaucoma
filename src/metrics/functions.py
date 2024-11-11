from typing import Tuple
import torch
from torch import Tensor
from typing import Optional, Union, List, Tuple
from torchmetrics.functional.classification.specificity_sensitivity import _convert_fpr_to_specificity
from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_arg_validation,
    _multiclass_precision_recall_curve_arg_validation,
    _multilabel_precision_recall_curve_arg_validation
)
from torchmetrics.functional.classification.roc import (
    _binary_roc_compute,
    _multiclass_roc_compute,
    _multilabel_roc_compute,
)

def _sensitivity_at_specificity(
    sensitivity: Tensor,
    specificity: Tensor,
    thresholds: Tensor,
    min_specificity: float,
) -> Tuple[Tensor, Tensor]:
    # get indices where specificity is greater than min_specificity
    indices = specificity >= min_specificity

    # if no indices are found, max_sens, best_threshold = 0.0, 1e6
    if not indices.any():
        max_sens = torch.tensor(0.0, device=sensitivity.device, dtype=sensitivity.dtype)
        best_threshold = torch.tensor(1e6, device=thresholds.device, dtype=thresholds.dtype)
    else:
        # redefine specificity, sensitivity, and threshold tensor based on indices
        specificity, sensitivity, thresholds = specificity[indices], sensitivity[indices], thresholds[indices]

        # get argmax
        idx = torch.argmax(sensitivity)

        # get max_sens and best_threshold
        max_sens, best_threshold = sensitivity[idx], thresholds[idx]

    return max_sens, best_threshold

def _binary_sensitivity_at_specificity_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    thresholds: Optional[Tensor],
    min_specificity: float,
    pos_label: int = 1,
) -> Tuple[Tensor, Tensor]:
    fpr, sensitivity, thresholds = _binary_roc_compute(state, thresholds, pos_label)
    specificity = _convert_fpr_to_specificity(fpr)
    return _sensitivity_at_specificity(sensitivity, specificity, thresholds, min_specificity)

def _binary_sensitivity_at_specificity_arg_validation(
    min_specificity: float,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
    if not isinstance(min_specificity, float) and not (0 <= min_specificity <= 1):
        raise ValueError(
            f"Expected argument `min_specificity` to be an float in the [0,1] range, but got {min_specificity}"
        )


def _multiclass_sensitivity_at_specificity_arg_validation(
    num_classes: int,
    min_specificity: float,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index)
    if not isinstance(min_specificity, float) and not (0 <= min_specificity <= 1):
        raise ValueError(
            f"Expected argument `min_specificity` to be an float in the [0,1] range, but got {min_specificity}"
        )

def _multiclass_sensitivity_at_specificity_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_classes: int,
    thresholds: Optional[Tensor],
    min_specificity: float,
) -> Tuple[Tensor, Tensor]:
    fpr, sensitivity, thresholds = _multiclass_roc_compute(state, num_classes, thresholds)
    specificity = [_convert_fpr_to_specificity(fpr_) for fpr_ in fpr]
    if isinstance(state, Tensor):
        res = [
            _sensitivity_at_specificity(sn, sp, thresholds, min_specificity)  # type: ignore
            for sn, sp in zip(sensitivity, specificity)
        ]
    else:
        res = [
            _sensitivity_at_specificity(sn, sp, t, min_specificity)
            for sn, sp, t in zip(sensitivity, specificity, thresholds)
        ]
    sensitivity = torch.stack([r[0] for r in res])
    thresholds = torch.stack([r[1] for r in res])
    return sensitivity, thresholds

def _multilabel_sensitivity_at_specificity_arg_validation(
    num_labels: int,
    min_specificity: float,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)
    if not isinstance(min_specificity, float) and not (0 <= min_specificity <= 1):
        raise ValueError(
            f"Expected argument `min_specificity` to be an float in the [0,1] range, but got {min_specificity}"
        )


def _multilabel_sensitivity_at_specificity_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_labels: int,
    thresholds: Optional[Tensor],
    ignore_index: Optional[int],
    min_specificity: float,
) -> Tuple[Tensor, Tensor]:
    fpr, sensitivity, thresholds = _multilabel_roc_compute(state, num_labels, thresholds, ignore_index)
    specificity = [_convert_fpr_to_specificity(fpr_) for fpr_ in fpr]
    if isinstance(state, Tensor):
        res = [
            _sensitivity_at_specificity(sn, sp, thresholds, min_specificity)  # type: ignore
            for sn, sp in zip(sensitivity, specificity)
        ]
    else:
        res = [
            _sensitivity_at_specificity(sn, sp, t, min_specificity)
            for sn, sp, t in zip(sensitivity, specificity, thresholds)
        ]
    sensitivity = torch.stack([r[0] for r in res])
    thresholds = torch.stack([r[1] for r in res])
    return sensitivity, thresholds