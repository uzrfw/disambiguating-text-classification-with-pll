from sklearn.feature_extraction.text import TfidfVectorizer

from encoder.encoder_type import EncoderType
from encoder.word_2_vec_vectorizer import Word2VecVectorizer


class Vectorizer:
    """
    Class for vectorization of text data.
    """

    def __init__(self, encoder_type: EncoderType, features: int = 100):
        """
        Initialize the Vectorizer object.

        :param: encoder_type (EncoderType): Type of encoder.
        :param: features (int, optional): Number of features. Defaults to 100.
        """
        self.encoder_type = encoder_type
        self.features = features

        if self.encoder_type == EncoderType.TFIDF:
            # Initialize TfidfVectorizer if encoder type is TFIDF
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=features)
        elif self.encoder_type == EncoderType.WORD_TO_VEC:
            # Initialize Word2VecVectorizer if encoder type is WORD_TO_VEC
            self.vectorizer = Word2VecVectorizer(features)

    def fit_transform(self, data):
        """
        Fit and transform the input data.

        :param: data: Input data to fit and transform.
        :returns: Transformed data.
        """
        return self.vectorizer.fit_transform(data)

    def transform(self, data):
        """
        Transform the input data.

        :param: data: Input data to transform.
        :returns: Transformed data.
        """
        return self.vectorizer.transform(data)
