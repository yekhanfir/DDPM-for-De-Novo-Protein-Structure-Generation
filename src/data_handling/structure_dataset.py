import torch
import pandas as pd
from data_utils import (
    make_np_example, 
    make_fixed_size, 
    center_positions,
    read_data_from_json,
)

class DatasetFromDataframe(torch.utils.data.Dataset):
    """Load coordinates data from a DataFrame, currently from the 'coords' column."""

    def __init__(self, data_config):
        self.dataframe, self.splits = self.get_data(
            data_config.data_file_path,
            data_config.splits_file_path
        )
        self.dataframe['split'] = self.dataframe.name.apply(
            lambda x: self.get_split(x)
        )
        self.dataframe['seq_len'] = self.dataframe.seq.apply(lambda x: len(x))

        self.max_seq_length = data_config.max_seq_length

    def get_data(self, data_file_path, splits_file_path):
        df, splits = read_data_from_json(
            self, 
            data_file_path, 
            splits_file_path
        )
        return df, splits

    def get_split(self, pdb_name):
        if pdb_name in self.splits.train[0]:
            return 'train'
        elif pdb_name in self.splits.validation[0]:
            return 'validation'
        elif pdb_name in self.splits.test[0]:
            return 'test'
        else:
            return 'None'


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords_dict = self.data.iloc[idx].coords
        np_example = make_np_example(coords_dict)
        make_fixed_size(np_example, self.max_seq_length)
        center_positions(np_example)
        example = {k: torch.tensor(v, dtype=torch.float32) for k, v
                   in np_example.items()}

        return example
    