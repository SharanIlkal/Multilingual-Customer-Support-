import torch
from torch.utils.data import Dataset

class CustomerSupportDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        label2idx,
        max_length=64,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row["text"]
        label = row["intent"]
        input_ids = self.tokenizer.encode(
            text,
            max_length=self.max_length,
        )
        label_id = self.label2idx[label]
        return (
            torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            torch.tensor(
                label_id,
                dtype=torch.long,
            ),
        )