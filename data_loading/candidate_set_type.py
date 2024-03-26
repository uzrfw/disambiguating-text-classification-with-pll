import enum


class CandidateSetType(enum.Enum):
    """
    Enumeration for different types of candidate sets
    """
    HEURISTIC = 'heuristic'
    LLAMA_1 = 'llama_one_label'
    LLAMA_2 = 'llama_two_labels'
    LLAMA_3 = 'llama_three_labels'
    MIX_HEURISTIC_LLAMA_1 = 'mix_heuristic_llama_one_labels'
    MIX_HEURISTIC_LLAMA_2 = 'mix_heuristic_llama_two_labels'

    def __str__(self):
        return self.name
