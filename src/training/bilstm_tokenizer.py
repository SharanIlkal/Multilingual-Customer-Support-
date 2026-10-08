import json
import re
from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

class SimpleTokenizer:

    def __init__(self, min_freq=1, max_vocab_size=50000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }
        self.idx2word = {
            0: PAD_TOKEN,
            1: UNK_TOKEN,
        }

    def tokenize(self, text):
        text = str(text).lower().strip()
        return re.findall(
            r"[^\W\d_]+(?:[^\W\d_]+)*|\d+",
            text,
            flags=re.UNICODE,
        )

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        words = [
            word
            for word, count in counter.most_common()
            if count >= self.min_freq
        ]
        words = words[: self.max_vocab_size - 2]
        for word in words:
            index = len(self.word2idx)
            self.word2idx[word] = index
            self.idx2word[index] = word

    def encode(
        self,
        text,
        max_length=64,
    ):

        tokens = self.tokenize(text)
        token_ids = [
            self.word2idx.get(token, self.word2idx[UNK_TOKEN],)
            for token in tokens
        ]
        token_ids = token_ids[:max_length]
        if len(token_ids) < max_length:
            token_ids += [self.word2idx[PAD_TOKEN]] * (max_length - len(token_ids))
        return token_ids

    def save(self, path):
        data = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path):
        tokenizer = cls()
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {
            int(k): v
            for k, v in data["idx2word"].items()
        }
        return tokenizer