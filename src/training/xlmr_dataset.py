import torch

from torch.utils.data import Dataset


class XLMRDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        label2idx,
        max_length=128,
    ):

        self.dataframe = (
            dataframe.reset_index(drop=True)
        )

        self.tokenizer = tokenizer

        self.label2idx = label2idx

        self.max_length = max_length

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        text = str(row["text"])

        label = row["intent"]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = (
            encoded["input_ids"]
            .squeeze(0)
        )

        attention_mask = (
            encoded["attention_mask"]
            .squeeze(0)
        )

        label_id = self.label2idx[
            label
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(
                label_id,
                dtype=torch.long,
            ),
        }