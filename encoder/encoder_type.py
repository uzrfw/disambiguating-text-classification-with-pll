import enum


class EncoderType(enum.Enum):
    # Enumeration class for different types of encoders

    TFIDF = 'tfidf'
    WORD_TO_VEC = 'word2vec'

    def __str__(self):
        return self.name
