import numpy as np

from data_loading.candidate_set_type import CandidateSetType
from encoder.encoder_type import EncoderType


class Dataset:
    """Class representing a dataset."""

    def __init__(
            self,
            x_full: np.ndarray,
            y_full: np.ndarray,
            y_true: np.ndarray,
            n_samples: int,
            m_features: int,
            l_classes: int,
            number_of_validation_data: int,
            encoder_type: EncoderType,
            candidate_set_type: CandidateSetType,
            features: int
    ) -> None:
        """
        Initialize Dataset object.

        :param: x_full (np.ndarray): Features of the dataset.
        :param: y_full (np.ndarray): Predicted labels.
        :param: y_true (np.ndarray): True labels.
        :param: n_samples (int): Total number of samples.
        :param: m_features (int): Number of features.
        :param: l_classes (int): Number of classes.
        :param: number_of_validation_data (int): Number of validation data points.
        :param: encoder_type (EncoderType): Type of encoder.
        :param: candidate_set_type (CandidateSetType): Type of candidate set.
        :param: features (int): Number of features.
        """
        self.x_full = x_full
        self.y_full = y_full
        self.y_true = y_true
        self.n_samples = int(n_samples)
        self.m_features = int(m_features)
        self.l_classes = int(l_classes)
        self.number_of_validation_data = int(number_of_validation_data)
        self.encoder_type = encoder_type
        self.candidate_set_type = candidate_set_type
        self.features = features

    def copy(self) -> "Dataset":
        """
        Create a copy of the dataset.

        :returns: A copy of the dataset.
        """

        return Dataset(
            self.x_full.copy(),
            self.y_full.copy(),
            self.y_true.copy(),
            self.n_samples,
            self.m_features,
            self.l_classes,
            self.number_of_validation_data,
            self.encoder_type,
            self.candidate_set_type,
            self.features
        )


class Datasplit:
    """Class representing a split of the dataset."""

    def __init__(
            self,
            x_train: np.ndarray,
            x_test: np.ndarray,
            y_train: np.ndarray,
            y_test: np.ndarray,
            y_true_train: np.ndarray,
            y_true_test: np.ndarray,
            orig_dataset: Dataset,
    ) -> None:
        """
        Initialize Datasplit object.

        :param: x_train (np.ndarray): Training features.
        :param: x_test (np.ndarray): Test features.
        :param: y_train (np.ndarray): Training predicted labels.
        :param: y_test (np.ndarray): Test predicted labels.
        :param: y_true_train (np.ndarray): True labels for training data.
        :param: y_true_test (np.ndarray): True labels for test data.
        :param: orig_dataset (Dataset): Original dataset.
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_true_train = y_true_train
        self.y_true_test = y_true_test
        self.orig_dataset = orig_dataset

    @classmethod
    def get_dataset(
            cls,
            dataset: Dataset
    ) -> "Datasplit":
        """
        Create a datasplit by splitting the data into training and test sets.

        :param: dataset (Dataset): Dataset to split.
        :returns: Datasplit: Split data.
        """

        return Datasplit(
            dataset.x_full[:-dataset.number_of_validation_data].copy(),
            dataset.x_full[-dataset.number_of_validation_data:].copy(),
            dataset.y_full[:-dataset.number_of_validation_data].copy(),
            dataset.y_full[-dataset.number_of_validation_data:].copy(),
            dataset.y_true[:-dataset.number_of_validation_data].copy(),
            dataset.y_true[-dataset.number_of_validation_data:].copy(),
            dataset,
        )

    def copy(self) -> "Datasplit":
        """
        Create a copy of the datasplit.

        :returns: A copy of the datasplit.
        """

        return Datasplit(
            self.x_train.copy(),
            self.x_test.copy(),
            self.y_train.copy(),
            self.y_test.copy(),
            self.y_true_train.copy(),
            self.y_true_test.copy(),
            self.orig_dataset.copy(),
        )
