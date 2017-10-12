"""UK WAC dataset."""
import gmtk

class UKWacDataset(gmtk.Dataset):
    def __init__(self, ukwac_path, result_path=None, training_set=True, test_set=True):
        """Instantiate UK Wac dataset from raw ukwac_path file (slow). Store and load
        resulting training examples to and from result_path (fast). Separates dataset
        into training and testing sets and builds vocabulary. Must specify whether to
        load training or test sets, cannot load both!

        Arguments:
            training_set - specify to load training set as dataset to train on
            test_set - specify to load test set as dataset to evaluate on"""
        assert (training_set and not test_set) or (test_set and not training_set)

        self.ukwac_path = ukwac_path

    def __getitem__(self, index):
        pass

    def __len__(self):
        return -1