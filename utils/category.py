import enum

from utils.category_type import CategoryType

# Define a list of specific category names
CATEGORY_NAMES_BIG: [str] = [
    "Literature",
    "Music",
    "Film",
    "Theater",
    "Visual Arts",
    "Philosophy",
    "Media",
    "Sport",
    "Medicine",
    "Food",
    "Drink",
    "History",
    "Biology",
    "Physics",
    "Chemistry",
    "Mathematics",
    "Geography",
    "Social Science",
    "Humanities",
    "Plant",
    "Nature",
    "Environment",
    "Waters",
    "Agriculture",
    "Biography",
    "Education",
    "Religion",
    "Society",
    "Leisure",
    "Politics",
    "Economics",
    "Business",
    "Finance",
    "Electronics",
    "Computer Science",
    "Engineering",
    "Transport",
    "Inventions",
    "Areas",
    "Places",
    "Buildings",
    "Animal",
]

# Define a list of general category names
CATEGORY_NAMES_MEDIUM: [str] = [
    "Culture and Art",
    "Health",
    "History",
    "Science",
    "People",
    "Religion",
    "Society",
    "Technology",
    "Geography and Places"
]


class Category(enum.Enum):
    LITERATURE = 1
    MUSIC = 2
    FILM = 3
    THEATER = 4
    VISUAL_ARTS = 5
    PHILOSOPHY = 6
    MEDIA = 7
    SPORT = 8
    MEDICINE = 9
    FOOD = 10
    DRINK = 11
    HISTORY = 12
    AGRICULTURE = 13
    BIOLOGY = 14
    PHYSICS = 15
    CHEMISTRY = 16
    MATHEMATICS = 17
    GEOGRAPHY = 18
    SOCIAL_SCIENCE = 19
    HUMANITIES = 20
    PLANT = 21
    NATURE = 22
    ENVIRONMENT = 23
    WATERS = 24
    BIOGRAPHY = 25
    EDUCATION = 26
    RELIGION = 27
    SOCIETY = 28
    LEISURE = 29
    POLITICS = 30
    ECONOMICS = 31
    BUSINESS = 32
    FINANCE = 33
    ELECTRONICS = 34
    COMPUTER_SCIENCE = 35
    ENGINEERING = 36
    TRANSPORT = 37
    INVENTIONS = 38
    AREAS = 39
    PLACES = 40
    BUILDINGS = 41
    ANIMAL = 42
    UNKNOWN = 2000


def find_category(category):
    """
    Returns the numeric value of a category.

    :param: category (str): The name of the category.
    :returns: int: The numeric value of the category, or 2000 (UNKNOWN) if not found.
    """
    try:
        return Category[category.upper()].value
    except KeyError:
        return Category.UNKNOWN.value


def get_category_names(category_type: CategoryType):
    """
    Returns a list of category names based on the category type.

    :param: category_type (CategoryType): The type of category (BIG or MEDIUM).
    :returns: list: List of category names.
    """
    if category_type == CategoryType.BIG:
        return CATEGORY_NAMES_BIG
    elif category_type == CategoryType.MEDIUM:
        return CATEGORY_NAMES_MEDIUM
    return []


def convert_categories_name_to_index(category_names: [str], category_type: CategoryType):
    """
    Converts a list of category names to their corresponding indices.

    :param: category_names ([str]): A list of category names.
    :param: category_type (CategoryType): The type of category classification (BIG or MEDIUM).
    :returns: [str]: List of category indices as strings.
    """
    category_list: [str] = CATEGORY_NAMES_BIG

    if category_type == CategoryType.MEDIUM:
        category_list = CATEGORY_NAMES_MEDIUM

    category_indexes: [str] = []

    for category_name in category_names:
        for index, element in enumerate(category_list):
            # Check if the category name is part of the element
            if category_name in element:
                category_indexes.append(str(index + 1))
                break

    return category_indexes


def convert_big_to_medium_category_text(big_category: str):
    """
    Convert a specific category name from big categories to medium categories.

    :param: big_category (str): The category name from big categories.
    :returns: str: The corresponding category name from medium categories.
    """
    if big_category == "Literature" or big_category == "Music" or big_category == "Film" or big_category == "Theater" or big_category == "Visual Arts" or big_category == "Philosophy" or big_category == "Media":
        return "Culture and Art"
    elif big_category == "Sport" or big_category == "Medicine" or big_category == "Food" or big_category == "Drink":
        return "Health"
    elif big_category == "History":
        return "History"
    elif big_category == "Biology" or big_category == "Animal" or big_category == "Physics" or big_category == "Chemistry" or big_category == "Mathematics" or big_category == "Geography" or big_category == "Social Science" or big_category == "Humanities" or big_category == "Plant" or big_category == "Nature" or big_category == "Environment" or big_category == "Waters" or big_category == "Agriculture":
        return "Science"
    elif big_category == "Biography" or big_category == "Education":
        return "People"
    elif big_category == "Religion":
        return "Religion"
    elif big_category == "Society" or big_category == "Leisure" or big_category == "Politics" or big_category == "Economics" or big_category == "Business" or big_category == "Finance":
        return "Society"
    elif big_category == "Electronics" or big_category == "Computer Science" or big_category == "Engineering" or big_category == "Transport" or big_category == "Inventions":
        return "Technology"
    elif big_category == "Areas" or big_category == "Places" or big_category == "Buildings":
        return "Geography and Places"


def convert_big_to_medium_category_index(big_category: str):
    """
    Convert a specific category name from big categories to medium categories.

    :param: big_category (str): The category name from big categories.
    :returns: str: The corresponding category name from medium categories.
    """
    big_category = str(big_category)
    if big_category == "0" or big_category == "1" or big_category == "2" or big_category == "3" or big_category == "4" or big_category == "5" or big_category == "6":
        return "1"
    elif big_category == "7" or big_category == "8" or big_category == "9" or big_category == "10":
        return "2"
    elif big_category == "11":
        return "3"
    elif big_category == "12" or big_category == "41" or big_category == "13" or big_category == "14" or big_category == "15" or big_category == "16" or big_category == "17" or big_category == "18" or big_category == "19" or big_category == "20" or big_category == "21" or big_category == "22" or big_category == "23":
        return "4"
    elif big_category == "24" or big_category == "25":
        return "5"
    elif big_category == "26":
        return "6"
    elif big_category == "27" or big_category == "28" or big_category == "29" or big_category == "30" or big_category == "31" or big_category == "32":
        return "7"
    elif big_category == "33" or big_category == "34" or big_category == "35" or big_category == "36" or big_category == "37":
        return "8"
    elif big_category == "38" or big_category == "39" or big_category == "40":
        return "9"

    print("ERROR" + big_category)
    return "0"


def convert_big_to_medium_category_index2(big_category: str):
    """
    Convert a specific category name from big categories to medium categories.

    :param: big_category (str): The category name from big categories.
    :returns: str: The corresponding category name from medium categories.
    """
    big_category = str(int(big_category) - 1)
    if big_category == "0" or big_category == "1" or big_category == "2" or big_category == "3" or big_category == "4" or big_category == "5" or big_category == "6":
        return "1"
    elif big_category == "7" or big_category == "8" or big_category == "9" or big_category == "10":
        return "2"
    elif big_category == "11":
        return "3"
    elif big_category == "12" or big_category == "41" or big_category == "13" or big_category == "14" or big_category == "15" or big_category == "16" or big_category == "17" or big_category == "18" or big_category == "19" or big_category == "20" or big_category == "21" or big_category == "22" or big_category == "23":
        return "4"
    elif big_category == "24" or big_category == "25":
        return "5"
    elif big_category == "26":
        return "6"
    elif big_category == "27" or big_category == "28" or big_category == "29" or big_category == "30" or big_category == "31" or big_category == "32":
        return "7"
    elif big_category == "33" or big_category == "34" or big_category == "35" or big_category == "36" or big_category == "37":
        return "8"
    elif big_category == "38" or big_category == "39" or big_category == "40":
        return "9"

    print("ERROR" + big_category)
    return "0"
