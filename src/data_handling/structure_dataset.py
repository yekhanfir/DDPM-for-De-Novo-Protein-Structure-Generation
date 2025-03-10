import pandas as pd
import torch
from data_handling.data_utils import (
    make_np_example, 
    make_fixed_size, 
    center_positions,
    read_local_data,
)
from datasets import load_dataset

class DatasetFromDataframe(torch.utils.data.Dataset):
    """Load coordinates data from a DataFrame, currently from the 'coords' column."""

    def __init__(self, data_path, data_source, split=None, max_seq_len=512):
        if data_source == "local":
            self.dataframe = read_local_data(data_path)
        else:
            print("Reading data from HF...")
            hf_dataset = load_dataset(data_path)[split]
            cols_to_remove = [
                col for col in hf_dataset.column_names if col not in [
                    "seq", "coords", 
                ]
            ]
            hf_dataset = hf_dataset.remove_columns(
                cols_to_remove
            )
            self.dataframe = pd.DataFrame(hf_dataset)
            print("Done.")

        self.dataframe['seq_len'] = self.dataframe.seq.apply(lambda x: len(x))

        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        coords_dict = self.dataframe.iloc[idx].coords
        np_example = make_np_example(coords_dict)
        make_fixed_size(np_example, self.max_seq_len)
        center_positions(np_example)
        example = {k: torch.tensor(v, dtype=torch.float32) for k, v
                   in np_example.items()}

        return example