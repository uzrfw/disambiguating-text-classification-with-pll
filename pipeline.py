import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import vstack

from data_loading.candidate_set_loader import CandidateSetLoader
from data_loading.candidate_set_type import CandidateSetType
from data_loading.data import Dataset
from data_loading.section_dataset import TextDataset
from data_loading.text_dataframe_loader import TextDataframeLoader
from encoder.encoder_type import EncoderType
from encoder.vectorizer import Vectorizer
from environment.env import getDataSourcePath
from experiments import (run_experiment, COLUMNS)
from utils.category import convert_big_to_medium_category_index

WIKIPEDIA_SECTIONS_TABLE = "wikipedia_sections"
WIKIPEDIA_SECTIONS_COLUMNS = ["sectionID", "cleanedArticleText"]

VALIDATION_TABLE = "validation_data"
VALIDATION_DATA_COLUMNS = ["sectionID", "sectionText", "class"]


class Pipeline:
    """
    A class representing a pipeline for running experiments on text datasets.
    """

    def start(self) -> None:
        """
        Starts the pipeline process.

        Loads and preprocesses Wikipedia sections data, prepares validation data,
        generates datasets, runs experiments, and saves results to a CSV file.
        """

        # Load and pre-process wikipedia sections
        loader_sections = TextDataframeLoader(getDataSourcePath(), WIKIPEDIA_SECTIONS_TABLE, WIKIPEDIA_SECTIONS_COLUMNS)
        train_data = loader_sections.get_all_data()
        text_lengths = np.array([len(text) for text in train_data[:, 1]])
        data = np.column_stack((train_data, text_lengths))

        lower_threshold = np.percentile(text_lengths, 10)
        upper_threshold = np.percentile(text_lengths, 90)
        train_data = data[(text_lengths > lower_threshold) & (text_lengths < upper_threshold)]

        # Load validation data
        loader_validation = TextDataframeLoader(getDataSourcePath(), VALIDATION_TABLE, VALIDATION_DATA_COLUMNS)
        validation_data = loader_validation.get_all_data()
        validation_ids, validation_x, validation_y = np.hsplit(validation_data, 3)
        validation_true_label = np.empty(validation_y.shape[0])

        for i in np.ndindex(validation_y.shape[0]):
            validation_true_label[i] = float(convert_big_to_medium_category_index(str(int(validation_y[i]) - 1))[0]) - 1

        # Create datasets
        train_dataset: TextDataset = TextDataset(train_data[:, 1], True)
        validation_dataset: TextDataset = TextDataset(validation_x, False)
        candidate_set_size_list = [10, 100, 500, 1000]

        # Run experiments
        resres = Parallel(n_jobs=1)(
            delayed(run_experiment)(dataset, seed, "pl-ecoc-2017")
            for seed in range(1)
            for dataset in generate_data_sets(
                train_dataset, validation_dataset, train_data, validation_data, validation_true_label,
                EncoderType.TFIDF, CandidateSetType.LLAMA_2, candidate_set_size_list
            )
        )

        res = sorted([row for res_row in resres for row in res_row])
        pd.DataFrame.from_records(res, columns=COLUMNS).to_csv("results/results.csv", index=False)


def generate_data_sets(train_dataset, validation_dataset, train_data, validation_data, validation_true_label,
                       encoder_type, candidate_set_type, candidate_set_size_list):
    """
    Generates a list of datasets for experimentation.

    :param: train_dataset (TextDataset): Dataset for training.
    :param: validation_dataset (TextDataset): Dataset for validation.
    :param: train_data (numpy.ndarray): Training data.
    :param: validation_data (numpy.ndarray): Validation data.
    :param: validation_true_label (numpy.ndarray): True labels for validation data.
    :param: encoder_type (EncoderType): Type of encoder for dataset encoding.
    :param: candidate_set_type (CandidateSetType): Type of candidate set.
    :param: candidate_set_size_list (list): List of candidate set sizes.

    :returns: list: List of generated datasets.
    """
    data_sets = []
    for candidate_set_size in candidate_set_size_list:
        data_set = generate_data_set(
            train_dataset, validation_dataset, train_data, validation_data, validation_true_label, encoder_type,
            candidate_set_type, candidate_set_size
        )
        data_sets.append(data_set)
    return data_sets


def generate_data_set(
        train_dataset,
        validation_dataset,
        train_data,
        validation_data,
        validation_true_label,
        encoder_type: EncoderType,
        candidate_set_type: CandidateSetType,
        features: int
):
    """
    Generates a dataset for experimentation.

    :param: train_dataset (TextDataset): Dataset for training.
    :param: validation_dataset (TextDataset): Dataset for validation.
    :param: train_data (numpy.ndarray): Training data.
    :param: validation_data (numpy.ndarray): Validation data.
    :param: validation_true_label (numpy.ndarray): True labels for validation data.
    :param: encoder_type (EncoderType): Type of encoder for dataset encoding.
    :param: candidate_set_type (CandidateSetType): Type of candidate set.
    :param: features (int): Number of features.

    :returns: Dataset: Generated dataset.
    """

    # Encode data
    vectorizer = Vectorizer(encoder_type, features)
    encoded_train_dataset = train_dataset.encode_data(vectorizer)
    encoded_validation_dataset = validation_dataset.encode_data(vectorizer)

    # Load candidate sets
    label_loader = CandidateSetLoader()
    candidate_set_data, candidate_set_validation = label_loader.get_candidate_set(candidate_set_type)

    df_section_ids = pd.DataFrame({'sectionID': train_data[:, 0]}).drop_duplicates(subset='sectionID')
    y_raw = pd.merge(df_section_ids, candidate_set_data, on='sectionID', how='left') \
        .fillna({'labels': '1,2,3,4,5,6,7,8,9'})

    df_v_section_ids = pd.DataFrame({'sectionID': validation_data[:, 0]}).drop_duplicates(subset='sectionID')
    v_y_raw = pd.merge(df_v_section_ids, candidate_set_validation, on='sectionID', how='left') \
        .fillna({'labels': '1,2,3,4,5,6,7,8,9'})

    # Extract cardinalities
    n_samples = encoded_train_dataset.shape[0] + encoded_validation_dataset.shape[0]
    m_features = encoded_train_dataset.shape[1]
    l_classes = 9

    # Partial label vector
    pl_vec = np.zeros((n_samples, l_classes), dtype=int)
    for i, y_val in y_raw.iloc[:, 1].items():
        if str(y_val) != 'nan':
            for j in str(y_val).split(','):
                pl_vec[i, int(j) - 1] = 1
    for i, y_val in v_y_raw.iloc[:, 1].items():
        if str(y_val) != 'nan':
            for j in str(y_val).split(','):
                pl_vec[i + encoded_train_dataset.shape[0], int(j) - 1] = 1

    # Create dataset
    if encoder_type == EncoderType.WORD_TO_VEC:
        dataset: Dataset = Dataset(
            np.vstack([encoded_train_dataset, encoded_validation_dataset]), pl_vec,
            np.concatenate([np.full(encoded_train_dataset.shape[0], 10), validation_true_label]),
            n_samples, m_features, l_classes, 1000, encoder_type, candidate_set_type, features
        )
    else:
        dataset: Dataset = Dataset(
            vstack([encoded_train_dataset, encoded_validation_dataset]), pl_vec,
            np.concatenate([np.full(encoded_train_dataset.shape[0], 10), validation_true_label]),
            n_samples, m_features, l_classes, 1000, encoder_type, candidate_set_type, features
        )

    return dataset
