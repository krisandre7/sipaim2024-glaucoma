from typing import Literal, Tuple, Type
from torch import Tensor
from typing import Optional, Union, List, Tuple, Any
from torchmetrics import Metric

from torchmetrics.utilities.data import dim_zero_cat as _cat
from torchmetrics.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)

from src.metrics.functions import (
    _binary_sensitivity_at_specificity_arg_validation,
    _binary_sensitivity_at_specificity_compute, 
    _multiclass_sensitivity_at_specificity_compute, 
    _multiclass_sensitivity_at_specificity_arg_validation,
    _multilabel_sensitivity_at_specificity_arg_validation,
    _multilabel_sensitivity_at_specificity_compute
)
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.utilities.enums import ClassificationTask

class BinarySensitivityAtSpecificity(BinaryPrecisionRecallCurve):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        min_specificity: float,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(thresholds, ignore_index, validate_args=False, **kwargs)
        if validate_args:
            _binary_sensitivity_at_specificity_arg_validation(min_specificity, thresholds, ignore_index)
        self.validate_args = validate_args
        self.min_specificity = min_specificity

    def compute(self) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        """Compute metric."""
        state = (_cat(self.preds), _cat(self.target)) if self.thresholds is None else self.confmat
        return _binary_sensitivity_at_specificity_compute(state, self.thresholds, self.min_specificity)
    
class MulticlassSensitivityAtSpecificity(MulticlassPrecisionRecallCurve):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def __init__(
        self,
        num_classes: int,
        min_specificity: float,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _multiclass_sensitivity_at_specificity_arg_validation(
                num_classes, min_specificity, thresholds, ignore_index
            )
        self.validate_args = validate_args
        self.min_specificity = min_specificity

    def compute(self) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        """Compute metric."""
        state = (_cat(self.preds), _cat(self.target)) if self.thresholds is None else self.confmat
        return _multiclass_sensitivity_at_specificity_compute(
            state, self.num_classes, self.thresholds, self.min_specificity
        )
        
class MultilabelSensitivityAtSpecificity(MultilabelPrecisionRecallCurve):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        min_specificity: float,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _multilabel_sensitivity_at_specificity_arg_validation(num_labels, min_specificity, thresholds, ignore_index)
        self.validate_args = validate_args
        self.min_specificity = min_specificity

    def compute(self) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        """Compute metric."""
        state = (_cat(self.preds), _cat(self.target)) if self.thresholds is None else self.confmat
        return _multilabel_sensitivity_at_specificity_compute(
            state, self.num_labels, self.thresholds, self.ignore_index, self.min_specificity
        )

class SensitivityAtSpecificity(_ClassificationTaskWrapper):
    def __new__(  # type: ignore[misc]
        cls: Type["SensitivityAtSpecificity"],
        task: Literal["binary", "multiclass", "multilabel"],
        min_specificity: float,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        if task == ClassificationTask.BINARY:
            return BinarySensitivityAtSpecificity(min_specificity, thresholds, ignore_index, validate_args, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassSensitivityAtSpecificity(
                num_classes, min_specificity, thresholds, ignore_index, validate_args, **kwargs
            )
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelSensitivityAtSpecificity(
                num_labels, min_specificity, thresholds, ignore_index, validate_args, **kwargs
            )
        raise ValueError(f"Task {task} not supported!")