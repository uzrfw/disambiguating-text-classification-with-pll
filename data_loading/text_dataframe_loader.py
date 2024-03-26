import sqlite3
from os import PathLike
from typing import List

import numpy as np
import pandas as pd


class TextDataframeLoader:
    def __init__(self, path_to_db: PathLike, table: str, columns: List[str]):
        """
        Initializes the TextDataframeLoader object.

        :param: path_to_db: Path to the SQLite database.
        :param: table: Name of the table to load data from.
        :param: columns: List of column names to include in the dataframe.
        """
        self.table = table
        self.columns = columns
        self.dataframe = self.load_dataframe_from_db(path_to_db, table, columns)

    def get_all_data(self) -> pd.DataFrame:
        """
         Returns the entire dataframe.

         :return: A pandas DataFrame containing all data.
         """
        return self.dataframe

    @staticmethod
    def load_dataframe_from_db(path_to_db: PathLike, table: str, columns_to_include: List[str]) -> pd.DataFrame:
        """
        Loads data from a SQLite database into a pandas DataFrame.

        :param: path_to_db: Path to the SQLite database.
        :param: table: Name of the table in the database.
        :param: columns_to_include: List of columns to include in the DataFrame.
        :return: A pandas DataFrame containing the selected data.
        """
        connection = sqlite3.connect(path_to_db)
        cursor = connection.cursor()

        columns = ", ".join(columns_to_include)
        query = "SELECT " + columns + " FROM " + table
        cursor.execute(query)
        data = cursor.fetchall()
        connection.close()

        sorted_data = sorted(data, key=lambda x: x[0])

        return np.array(sorted_data)
