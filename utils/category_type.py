import enum


class CategoryType(enum.Enum):
    """
    Enumeration for different types of categories.
    """
    BIG = 'big',
    MEDIUM = 'medium'

    def __str__(self):
        return self.name
