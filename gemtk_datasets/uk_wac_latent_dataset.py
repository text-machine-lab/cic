"""Dataset constructed by passing every message in the UK Wac dataset through
a pre-trained autoencoder."""
from arcadian.dataset import Dataset
from cic.gemtk_datasets.uk_wac_dataset import UKWacDataset

class LatentUKWacDataset(Dataset):
    def __init__(self, ukwac, autoencoder):
        self.np_l_messages = autoencoder.predict(ukwac, output_tensors=['code'])['code']

    def __getitem__(self, index):
        # Get a single item as an index from the dataset.
        return {'code': self.np_l_messages[index, :]}

    def __len__(self):
        # Return the length of the dataset.
        return len(self.np_l_messages)