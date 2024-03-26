from typing import Optional

import numpy as np


class SplitResult:
    """
    Results on either the train or test set.

    This code was imported from the repository https://github.com/anon1248/uncertainty-aware-pll
    """

    def __init__(
            self, pred: np.ndarray,
            is_sure_pred: np.ndarray,
            is_guessing: np.ndarray,
    ) -> None:
        self.pred = pred
        self.is_sure_pred = is_sure_pred
        self.is_guessing = is_guessing

    def frac_sure_predictions(self) -> float:
        """ Returns the fraction of sure predictions.

        Returns:
            float: The fraction of sure predictions.
        """

        return np.count_nonzero(self.is_sure_pred) / self.is_sure_pred.shape[0]

    def frac_guessing(self) -> float:
        """ Returns the fraction guessing.

        Returns:
            float: The fraction of guessing.
        """

        return np.count_nonzero(self.is_guessing) / self.is_guessing.shape[0]

    def sure_predictions(self) -> np.ndarray:
        """ Returns the sure predictions.

        Returns:
            np.ndarray: The sure predictions.
        """

        return self.pred[self.is_sure_pred]


class Result:
    """
    Results on train and test set.

    This code was imported from the repository https://github.com/anon1248/uncertainty-aware-pll
    """

    def __init__(self, test_result: Optional[SplitResult]) -> None:
        self.test_result = test_result

    def is_transductive_setting(self) -> bool:
        """ Whether we are in a transductive setting.

        Returns:
            bool: Whether transductive.
        """

        return self.test_result is None

    def get_test_result(self) -> SplitResult:
        """ Safely returns the test results.

        Raises:
            ValueError: If results do not exist.

        Returns:
            SplitResult: The test results.
        """

        if self.test_result is None:
            raise ValueError("Result must exist.")
        return self.test_result
