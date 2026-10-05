from torch.utils.data import Dataset
import torch


class LanguageModelDataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer,
        seq_len: int,
    ):
        self.seq_len = seq_len

        tokens = []

        for text in texts:
            ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            tokens.extend(ids)
            tokens.append(tokenizer.eos_token_id)

        self.tokens = torch.tensor(
            tokens,
            dtype=torch.long,
        )

    def __len__(self):
        # Need seq_len + 1 tokens because targets are shifted by one
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len

        chunk = self.tokens[start : start + self.seq_len + 1]

        input_ids = chunk[:-1]
        targets = chunk[1:]

        return input_ids, targets
