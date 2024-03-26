import sqlite3

import pandas as pd

from data_loading.candidate_set_type import CandidateSetType
from environment.env import getDataSourcePath


class CandidateSetLoader:
    """Class to load candidate sets from SQLite database."""

    def __init__(self):
        """Initialize CandidateSetLoader."""

        # Establish connection to SQLite database
        db_path = getDataSourcePath()
        connection = sqlite3.connect(db_path)

        # Load dataframes from SQLite tables
        self.df_heuristics = pd.read_sql_query(
            'SELECT sectionID, GROUP_CONCAT(label) AS labels FROM candidate_set_heuristics_training GROUP BY sectionID',
            connection)
        self.df_heuristics_validation = pd.read_sql_query(
            'SELECT sectionID, GROUP_CONCAT(label) AS labels FROM candidate_set_heuristics_validation GROUP BY sectionID',
            connection)

        self.df_lama2_1 = pd.read_sql_query(
            'SELECT sectionID, GROUP_CONCAT(label) AS labels FROM candidate_set_llama2_1_training GROUP BY sectionID',
            connection)
        self.df_lama2_1_validation = pd.read_sql_query(
            'SELECT sectionID, GROUP_CONCAT(label) AS labels FROM candidate_set_llama2_1_validation GROUP BY sectionID',
            connection)

        self.df_lama2_2 = pd.read_sql_query(
            'SELECT sectionID, GROUP_CONCAT(label) AS labels FROM candidate_set_llama2_2_training GROUP BY sectionID',
            connection)
        self.df_lama2_2_validation = pd.read_sql_query(
            'SELECT sectionID, GROUP_CONCAT(label) AS labels FROM candidate_set_llama2_2_validation GROUP BY sectionID',
            connection)

        self.df_lama2_3 = pd.read_sql_query(
            'SELECT sectionID, GROUP_CONCAT(label) AS labels FROM candidate_set_llama2_3_training GROUP BY sectionID',
            connection)
        self.df_lama2_3_validation = pd.read_sql_query(
            'SELECT sectionID, GROUP_CONCAT(label) AS labels FROM candidate_set_llama2_3_validation GROUP BY sectionID',
            connection)

    def get_candidate_set(self, candidate_set_type: CandidateSetType):
        """
        Retrieve candidate sets based on the provided CandidateSetType.

        :param: candidate_set_type (CandidateSetType): Type of candidate set to retrieve.
        :returns: A tuple containing two pandas DataFrames representing the candidate sets.
        """
        if candidate_set_type == CandidateSetType.LLAMA_1:
            return self.df_lama2_1, self.df_lama2_1_validation

        if candidate_set_type == CandidateSetType.LLAMA_2:
            return self.df_lama2_2, self.df_lama2_2_validation

        if candidate_set_type == CandidateSetType.LLAMA_3:
            return self.df_lama2_3, self.df_lama2_3_validation

        elif candidate_set_type == CandidateSetType.HEURISTIC:
            return self.df_heuristics, self.df_heuristics_validation

        elif candidate_set_type == CandidateSetType.MIX_HEURISTIC_LLAMA_1:
            df_section_ids = pd.merge(pd.DataFrame({'sectionID': self.df_heuristics['sectionID']}),
                                      pd.DataFrame({'sectionID': self.df_lama2_1['sectionID']}),
                                      how='outer').drop_duplicates(subset='sectionID')
            df_mix = pd.merge(df_section_ids, self.df_lama2_1, on='sectionID', how='left')
            df_mix = pd.merge(df_mix, self.df_heuristics, on='sectionID', how='left') \
                .fillna({'labels': '1,2,3,4,5,6,7,8,9'})

            df_v_section_ids = pd.merge(pd.DataFrame({'sectionID': self.df_heuristics_validation['sectionID']}),
                                        pd.DataFrame({'sectionID': self.df_lama2_1_validation['sectionID']}),
                                        how='outer').drop_duplicates(subset='sectionID')
            df_v_mix = pd.merge(df_v_section_ids, self.df_lama2_1_validation, on='sectionID', how='left')
            df_v_mix = pd.merge(df_v_mix, self.df_heuristics_validation, on='sectionID', how='left') \
                .fillna({'labels': '1,2,3,4,5,6,7,8,9'})

            return df_mix, df_v_mix

        elif candidate_set_type == CandidateSetType.MIX_HEURISTIC_LLAMA_2:
            df_section_ids = pd.merge(pd.DataFrame({'sectionID': self.df_heuristics['sectionID']}),
                                      pd.DataFrame({'sectionID': self.df_lama2_2['sectionID']}),
                                      how='outer').drop_duplicates(subset='sectionID')
            df_mix = pd.merge(df_section_ids, self.df_lama2_2, on='sectionID', how='left')
            df_mix = pd.merge(df_mix, self.df_heuristics, on='sectionID', how='left') \
                .fillna({'labels': '1,2,3,4,5,6,7,8,9'})

            df_v_section_ids = pd.merge(pd.DataFrame({'sectionID': self.df_heuristics_validation['sectionID']}),
                                        pd.DataFrame({'sectionID': self.df_lama2_2_validation['sectionID']}),
                                        how='outer').drop_duplicates(subset='sectionID')
            df_v_mix = pd.merge(df_v_section_ids, self.df_lama2_2_validation, on='sectionID', how='left')
            df_v_mix = pd.merge(df_v_mix, self.df_heuristics_validation, on='sectionID', how='left') \
                .fillna({'labels': '1,2,3,4,5,6,7,8,9'})

            return df_mix, df_v_mix
