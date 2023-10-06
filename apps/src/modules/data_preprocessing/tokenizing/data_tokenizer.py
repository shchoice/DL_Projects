import pandas as pd
from konlpy.tag import Mecab


class DataTokenizer:
    def __init__(self, data_config):
        self.data_config = data_config

        if self.tokenier_type == "mecab":
            self.tokenizer = Mecab()

    def tokenize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.tokenizer_type == "whitespace":
            df['Tokenized_Text'] = df['Text'].str.split()
        elif self.tokenizer_type == "mecab":
            df['Tokenized_Text'] = df['Text'].apply(self.tokenizer.morphs)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")
        return df
