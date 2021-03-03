import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import default_collate
import csv
from collections import defaultdict

class Vocab(defaultdict):
    def __init__(self, train=True):
        super().__init__(lambda : len(self))
        self.train = train
        self.UNK = "UNK"
        # set UNK token to 0 index
        self[self.UNK]

    def eval(self):
        self.train = False

    def train(self):
        self.train = True

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return [self[w] for w in ws]
        else:
            return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if int(i) in idx2w else "UNK" for i in ids]

class Split(object):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_words(self, ws):
        return pack_sequence(ws)

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda item : len(item[0]), reverse=True)
        words = pack_sequence([w for w,_ in batch])
        targets = default_collate([t for _,t in batch])
        return words, targets


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, vocab, lower_case=True):
        # load the data
        labels, texts = [], []
        with open(data_file) as infile:
            reader = csv.reader(infile, delimiter="\t")
            for label, text in reader:
                labels.append(label)
                if lower_case:
                    texts.append(text.lower())
                else:
                    texts.append(text)
        # Convert labels to numeric form
        label_map = {"Positive": 1, "Negative": 0}
        labels = [torch.LongTensor([label_map[label]]) for label in labels]

        # Convert text to vector form
        texts = [torch.LongTensor(vocab.ws2ids(text.split())) for text in texts]
        self.data = Split([(text, label) for text, label in zip(texts, labels)])

    def get_split(self):
        return Split(self.data)



