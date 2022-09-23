import re

import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer


class ClassificationDataset:
    def __init__(self, dataframe, model_config, max_pad_len):
        translate_label = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 9: 7, 10: 8}
        self.max_len = max_pad_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_config, do_lower_case=True)

        self.labels = [translate_label[label] for label in dataframe['groups']]
        self.tokens = [self.get_token(text) for text in tqdm(dataframe['name'], total=len(dataframe))]

    @staticmethod
    def regular_clear(text):
        text = text.lower()
        without_extra = text.replace('/', ' ').replace(':', ' ')
        without_brackets = re.sub('[\(\[].*?[\)\]]', '', without_extra)
        without_quoted_text = re.sub('".*?"', '', without_brackets)
        without_quoted_text = re.sub("'.*?'", '', without_quoted_text)
        without_latin = re.sub(r'[a-zA-Z_]', r'', without_quoted_text)
        without_double_spaces = re.sub("\s\s+", " ", without_latin)

        without_long_digits = ' '.join([line_piece for line_piece in without_double_spaces.split(' ') if
                                        not (line_piece.isdigit() and len(line_piece) > 3)])

        return without_long_digits.strip()

    def get_token(self, text):
        clear_text = self.regular_clear(text)
        tokens = self.tokenizer(
            clear_text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )

        return tokens

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.tokens[item], self.labels[item]


class ClassificationDatasetInfer:
    def __init__(self, dataframe):
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", do_lower_case=True)

        self.tokens = [self.get_token(text) for text in tqdm(dataframe['name'], total=len(dataframe)) if not pd.isna(text)]

    def get_token(self, text):
        tokens = self.tokenizer(
            text,
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )

        return tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item]
