"""UK WAC dataset."""
from gemtk_datasets import string_dataset


class UKWacDataset(string_dataset.StringDataset):
    def __init__(self, ukwac_path, result_path=None, token_to_id=None, max_length=None, training_set=False, test_set=False):
        """Instantiate UK Wac dataset from raw ukwac_path file (slow). Store and load
        resulting training examples to and from result_path (fast). Separates dataset
        into training and testing sets and builds vocabulary. Must specify whether to
        load training or test sets, cannot load both!

        Arguments:
            training_set - specify to load training set as dataset to train on
            test_set - specify to load test set as dataset to evaluate on"""
        assert (training_set and not test_set) or (test_set and not training_set)

        self.ukwac_path = ukwac_path

        def contains_numbers(s):
            return any(c.isdigit() for c in s)

        # Read all sentences from file, and keep them if they follow the correct formatting.
        filtered_sentences = []
        for index, line in enumerate(open(self.ukwac_path, 'r', encoding='utf-8', errors='ignore')):
            sentences = line.split(' . ')
            for each_sentence in sentences:
                each_sentence = each_sentence.lower()
                #print(each_sentence)
                if not each_sentence.isspace() \
                        and '(' not in each_sentence \
                        and ')' not in each_sentence \
                        and not each_sentence.startswith('current url') \
                        and not contains_numbers(each_sentence) \
                        and '"' not in each_sentence \
                        and ':' not in each_sentence:
                    each_sentence += '.'
                    each_sentence = each_sentence.strip()
                    if max_length is None or len(each_sentence.split()) < max_length:
                        filtered_sentences.append(each_sentence)

        print('Number of filtered sentences: %s' % len(filtered_sentences))

        # Use StringDataset class to automatically tokenize and convert strings to numpy format.
        super().__init__(filtered_sentences, max_length=max_length, token_to_id=token_to_id)

        for index in range(10):
            print(self.formatted_and_filtered_strings[index])
            print(self.np_messages[index])

if __name__ == '__main__':
    # Example driver program for constructing a UKWacDataset object.
    ukwac_path = '/data2/arogers/Corpora/En/UkWac/Plain-txt/ukwac_subset_100M.txt'
    result_path = './data/ukwac/'
    print('Loading dataset...')
    ukwac = UKWacDataset(ukwac_path, result_path=result_path, max_length=20, training_set=True)
    print('Number of numpy messages in dataset: %s' % ukwac.np_messages.shape[0])