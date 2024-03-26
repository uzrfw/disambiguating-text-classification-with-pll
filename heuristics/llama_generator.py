from utils.category import get_category_names, convert_categories_name_to_index
from utils.category_type import CategoryType
from utils.class_type import ClassType
from utils.llama2_requests import request_llama2, extract_categories_from_response, \
    extract_categories_from_response_ordered

LLAMA_2_PROMPT_SINGLE: str = "Given the text snippet: %s, please analyze its content and categorize it under the most fitting label from the following options: %s. Provide the best matching category based on the content of the text."
LLAMA_2_PROMPT_MULTIPLE: str = "Given the text snippet: %s, please analyze its content and categorize it under the three most fitting labels from the following options: %s. Provide the best three matching categories based on the content of the text."
LLAMA_2_PROMPT_ORDER: str = "Given the text snippet: %s, please analyze its content and categorize it under the most fitting labels from the following options: Culture and Art, Health and Sports, History, Science, People and Education, Religion, Society, Technology, Geography and Places. Provide a comprehensive list of these options starting with the most relevant and descending in order of relevance based on the content of the text."


class LlamaGenerator:
    """
    Class for generating labels using LLama2.
    """

    def __init__(self, category_type: CategoryType, class_type: ClassType):
        """
        Initialize the LlamaGenerator object.

        :param: category_type (CategoryType): Type of category.
        :param: class_type (ClassType): Type of class.
        """
        self._category_type = category_type
        self._class_type = class_type
        self.categories: [str] = get_category_names(category_type)
        self.category_string = convert_categories_to_string(self.categories)

    def get_llama_labels(self, text: str):
        """
        Predict the class of a text with LLama2

        :param: text (str): text to classify
        :return: list: List of possible categories
        """
        prompt: str

        # Create a prompt based on the class type
        if self._class_type == ClassType.SINGLE:
            prompt = LLAMA_2_PROMPT_SINGLE % (text, self.category_string)
        elif self._class_type == ClassType.MULTIPLE:
            prompt = LLAMA_2_PROMPT_MULTIPLE % (text, self.category_string)
        else:
            prompt = LLAMA_2_PROMPT_ORDER % text

        print(prompt)
        # Create a prompt based on the class type
        llama_response = request_llama2(prompt)
        print("RESPONSE:" + llama_response)

        # Create a prompt based on the class type
        if self._class_type == ClassType.ORDER:
            extracted_categories = extract_categories_from_response_ordered(self.categories, llama_response)
        else:
            extracted_categories = extract_categories_from_response(self.categories, llama_response)

        if self._class_type == ClassType.SINGLE:
            extracted_categories = extracted_categories[:1]
        elif self._class_type == ClassType.MULTIPLE:
            extracted_categories = extracted_categories[:3]
        # Convert the extracted categories to their indices
        return convert_categories_name_to_index(extracted_categories, self._category_type)


def convert_categories_to_string(categories: [str]):
    """
    Convert a list of categories to a comma-separated string.

    :param: categories (list): List of category names.
    :return: str: Comma-separated string representing the categories.
    """
    result: str = ""
    for category in categories:
        result += "" + category + ","

    return result
