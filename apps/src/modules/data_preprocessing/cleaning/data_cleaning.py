from typing import List

import pandas as pd


class DataCleaning:
    def __init__(self, data_config):
        self.data_config = data_config

    def clean_df(self, df: pd.DataFrame, df_columns: List[str]):
        df = self.delete_null_data(df)
        df = self.strip_column(df, df_columns)
        df = self.delete_duplicate_data(df, df_columns[1:3])

        return df

    @staticmethod
    def delete_null_data(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=0)

    @staticmethod
    def strip_column(df: pd.DataFrame, df_columns: List[str]) -> pd.DataFrame:
        for df_column in df_columns:
            df[df_column] = df[df_column].astype(str).str.strip()

        return df

    @staticmethod
    def delete_duplicate_data(df: pd.DataFrame, df_columns: List[str]) -> pd.DataFrame:
        return df.drop_duplicates(df_columns, keep='first', inplace=True)
