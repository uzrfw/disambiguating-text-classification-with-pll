""" Module for constant classifier. """
from typing import Optional

import numpy as np
from data_loading.data import Datasplit

from result import SplitResult


class ConstantClassifier:
    """ Constant classifier. """

    def __init__(self, data: Datasplit, constant_val: float = 0.0) -> None:
        self.data = data
        self.constant_val = constant_val

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """ Returns constant values.

        Args:
            data (np.ndarray): The data specifying the shape.

        Returns:
            np.ndarray: Constant values.
        """

        return self.constant_val * np.ones(data.shape[0])

    def get_train_pred(self) -> SplitResult:
        """ Get the label predictions on the training set.

        Returns:
            SplitResult: The predictions.
        """

        return SplitResult(
            pred=np.zeros(len(self.data.y_test), dtype=bool),
            is_sure_pred=np.zeros(len(self.data.y_test), dtype=bool),
            is_guessing=np.ones(len(self.data.y_test), dtype=bool),
        )

    def get_test_pred(self) -> Optional[SplitResult]:
        """ Get the label predictions on the test set.

        Returns:
            Optional[SplitResult]: The predictions.
        """

        if self.data.x_test.shape[0] == 0:
            return None

        y_test_pred_list = []
        for _ in self.data.y_test:
            y_test_pred_list.append(self.constant_val)

        return SplitResult(
            pred=np.array(y_test_pred_list),
            is_sure_pred=np.zeros(len(y_test_pred_list), dtype=bool),
            is_guessing=np.ones(len(y_test_pred_list), dtype=bool),
        )
