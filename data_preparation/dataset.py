from torch.utils.data import Dataset


class TripletDataset(Dataset):

    def __init__(self, data, labels=None):
        self.input_ids = data[0]
        self.attention_mask = data[1]
        self.labels = labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]
        else:
            return self.input_ids[idx], self.attention_mask[idx]
