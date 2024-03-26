import enum


class ClassType(enum.Enum):
    """
    Enumeration for different types of classes
    """
    SINGLE = 'single'
    MULTIPLE = 'multiple'
    ORDER = 'order'

    def __str__(self):
        return self.name
