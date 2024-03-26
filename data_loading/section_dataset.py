import pandas as pd


class TextDataset:
    def __init__(
            self,
            data: pd.DataFrame,
            is_training_set: bool
    ):
        """
        Initializes the TextDataset object.

        :param: data: A pandas DataFrame containing the dataset.
        :param: is_training_set: Boolean flag to indicate if this dataset is for training purposes.
        """
        self.data = data
        self.is_training_set = is_training_set

    def encode_data(self, vectorizer):
        """
        Encodes the dataset based on the specified encoder type.

        :return: Encoded data.
        """

        if self.is_training_set:
            encoded_data = vectorizer.fit_transform(self.data.flatten())
        else:
            encoded_data = vectorizer.transform(self.data.flatten())

        return encoded_data
