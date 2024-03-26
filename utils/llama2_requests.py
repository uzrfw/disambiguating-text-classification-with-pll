import requests

from environment.env import getLlama2URL


def request_llama2(prompt: str):
    """
    Sends a request to the llama2 API with a given prompt.

    :param prompt: The prompt to send to the llama2 API.
    :return str: The text response from the llama2 API.
    """
    headers = {}
    data = "{\"prompt\":\"<s>[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\n\\n%s [/INST]\\n\",\"model\":\"meta/llama-2-70b-chat\",\"systemPrompt\":\"You are a helpful assistant.\",\"temperature\":0.75,\"topP\":0.9,\"maxTokens\":800,\"image\":null,\"audio\":null}" % prompt
    response = requests.request("POST", getLlama2URL(), headers=headers, data=data)
    return response.text


def extract_categories_from_response(categories: [str], response: str):
    """
    Extracts specified categories from a response.

    :param categories: A list of categories to look for in the response.
    :param response: The text response.
    :returns: A list of categories that were found in the response.
      """

    contained_categories: [str] = []

    for category in categories:
        if category in response:
            # If category is in response, add the category to the list
            contained_categories.append(category)

    return contained_categories


def extract_categories_from_response_ordered(categories: [str], response: str):
    """
    Extracts specified categories from a response.

    :param categories: A list of categories to look for in the response.
    :param response: The text response.
    :returns: A list of categories that were found in the response.
      """

    contained_categories: [str] = []

    cat: [str] = [
        ["Culture and Art", "Culture"],
        ["Health", "Health and Sports", "health", "Sport", "sport"],
        ["History"],
        ["Science"],
        ["People", "People and Education", "Education"],
        ["Religion"],
        ["Society", "Politics"],
        ["Technology"],
        ["Geography and Places", "Geography", "Places"]
    ]

    for category_list in cat:
        contains: bool = False
        index = -1
        for category in category_list:
            if category in response:
                index = response.index(category)
                contains = True
                break

        if contains:
            # Add category along with its index position to the list
            contained_categories.append((category_list[0], index))

    # Sort the list based on the index positions
    contained_categories.sort(key=lambda x: x[1])

    # Return only the categories in the order they appear
    return [category[0] for category in contained_categories]
