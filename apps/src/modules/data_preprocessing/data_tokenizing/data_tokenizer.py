import pandas as pd
from konlpy.tag import Mecab


class DataTokenizer:
    def __init__(self, data_config):
        self.data_config = data_config
        self.tokenizer = None

        if self.data_config['tokenizer_type'] == "mecab":
            self.tokenizer = Mecab()

    def tokenize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.data_config['tokenizer_type'] == "whitespace":
            df['Text'] = df['Text'].str.split()
        elif self.data_config['tokenizer_type'] == "mecab":
            df['Text'] = df['Text'].apply(self.tokenizer.morphs)
        elif self.data_config['tokenizer_type'] == "None" or self.data_config['tokenizer_type'] is None:
            return df
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.data_config['tokenizer_type']}")
        return df
