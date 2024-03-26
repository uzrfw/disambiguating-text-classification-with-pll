import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

nltk.download('punkt')


class Word2VecVectorizer:
    """
    Class for vectorizing text data using Word2Vec embeddings.
    """

    def __init__(self, features=100):
        """
        Initialize the Word2VecVectorizer object.

        :param: features (int, optional): Number of features. Defaults to 100.
        """
        self.features = features
        self.word2vec_model = None

    def _average_word_vectors(self, words, model, num_features):
        """
        Calculate the average word vectors.

        :param: words (list): List of words.
        :param: model: Word2Vec model.
        :param: num_features (int): Number of features.

        :returns: np.ndarray: Average word vector.
        """
        feature_vector = np.zeros((num_features,), dtype="float64")
        num_words = 0
        for word in words:
            if word in model.wv:
                num_words += 1
                feature_vector = np.add(feature_vector, model.wv[word])

        if num_words > 0:
            feature_vector = np.divide(feature_vector, num_words)
        return feature_vector

    def fit_transform(self, data):
        """
        Fit the Word2Vec model and transform the input data.

        :param: data: Input data.
        :returns: np.ndarray: Transformed data.
        """
        tokenized_data = [word_tokenize(doc.lower()) for doc in data.flatten()]

        self.word2vec_model = Word2Vec(sentences=tokenized_data, vector_size=self.features, window=5, min_count=1,
                                       workers=4)

        encoded_data = [self._average_word_vectors(words, self.word2vec_model, self.features) for words in
                        tokenized_data]
        encoded_data = np.vstack(encoded_data)

        return encoded_data

    def transform(self, data):
        """
        Transform the input data using the trained Word2Vec model.

        :param: data: Input data.
        :returns: np.ndarray: Transformed data.
        """
        tokenized_data = [word_tokenize(doc.lower()) for doc in data.flatten()]

        if self.word2vec_model is None:
            raise ValueError("Train first")

        encoded_data = [self._average_word_vectors(words, self.word2vec_model, self.features) for words in
                        tokenized_data]
        encoded_data = np.vstack(encoded_data)

        return encoded_data
