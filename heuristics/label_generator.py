from heuristics.heuristic_generator import HeuristicGenerator
from heuristics.llama_generator import LlamaGenerator
from utils.category_type import CategoryType
from utils.class_type import ClassType


class LabelGenerator:
    """
    Class for generating labels for text classification.
    """

    def __init__(self, category_type: CategoryType, class_type: ClassType):
        """
        Initialize the LabelGenerator object.

        :param: category_type (CategoryType): Type of category.
        :param: class_type (ClassType): Type of class.
        """
        self._category_type = category_type
        self._class_type = class_type
        self.llama_generator = LlamaGenerator(category_type, class_type)
        self.heuristic_generator = HeuristicGenerator(category_type)

    def get_labels(self, text: str):
        """
        Predicts the class of a text with LLama2 and heuristics

        :param: text (str): text to classify
        :return: list: List of possible categories
        """
        llama_labels = self.llama_generator.get_llama_labels(text)
        heuristic_labels = self.heuristic_generator.get_heuristic_labels(text)

        labels = list(set(llama_labels + heuristic_labels))
        return labels
