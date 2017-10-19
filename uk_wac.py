"""UK WAC dataset."""
import gmtk

class UKWacDataset(gmtk.Dataset):
    def __init__(self, ukwac_path, result_path=None, training_set=False, test_set=False):
        """Instantiate UK Wac dataset from raw ukwac_path file (slow). Store and load
        resulting training examples to and from result_path (fast). Separates dataset
        into training and testing sets and builds vocabulary. Must specify whether to
        load training or test sets, cannot load both!

        Arguments:
            training_set - specify to load training set as dataset to train on
            test_set - specify to load test set as dataset to evaluate on"""
        assert (training_set and not test_set) or (test_set and not training_set)

        self.ukwac_path = ukwac_path

        filtered_sentences = []
        for index, line in enumerate(open(self.ukwac_path, 'r', encoding='utf-8', errors='ignore')):
            if index < 4:
                # print(line)
                sentences = line.split(' . ')
                for each_sentence in sentences:
                    each_sentence = each_sentence.lower()
                    #print(each_sentence)
                    if not each_sentence.isspace() \
                            and '(' not in each_sentence \
                            and ')' not in each_sentence \
                            and not each_sentence.startswith('current url'):
                        each_sentence += '.'
                        each_sentence = each_sentence.strip()
                        filtered_sentences.append(each_sentence)
            else:
                break

        for filtered_sentence in filtered_sentences:
            print(filtered_sentence)

    def __getitem__(self, index):
        pass

    def __len__(self):
        return -1


if __name__ == '__main__':
    ukwac_path = '/data2/arogers/Corpora/En/UkWac/Plain-txt/ukwac_subset_100M.txt'
    result_path = './data/ukwac/'
    print('Loading dataset...')
    ukwac = UKWacDataset(ukwac_path, result_path=result_path, training_set=True)