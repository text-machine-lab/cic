"""Tests for TorontoBookCorpus dataset class."""
import unittest2
from cic.datasets.book_corpus import TorontoBookCorpus
from cic.datasets.text_dataset import convert_numpy_array_to_strings
import cic.paths

class TorontoBookCorpusTest(unittest2.TestCase):
    def test_empty(self):
        print('Hello World')

    def test_construction(self):
        tbc = TorontoBookCorpus(20, result_path=cic.paths.BOOK_CORPUS_RESULT,
                                min_length=5, max_part_len=10, max_num_s=100)

        print(tbc.vocab)

        print('\n\n')

        inverse_vocab = {tbc.vocab[k]:k for k in tbc.vocab}

        messages = convert_numpy_array_to_strings(tbc.data, inverse_vocab,
                                                  tbc.stop_token,
                                                  keep_stop_token=False)

        print('Number of examples: %s' % len(tbc))

        with open(cic.paths.BOOK_CORPUS_P1) as file:
            for index, line in enumerate(file):

                if index >= len(tbc):
                    break

                print(line.replace('\n', ''))
                print(messages[index])
                print()

    def test_time(self):
        print()
        print('Starting dataset creation')
        tbc = TorontoBookCorpus(20, result_path=cic.paths.BOOK_CORPUS_RESULT,
                                min_length=5, max_num_s=1000000, keep_unk_sentences=False,
                                vocab_min_freq=5)
        print('Finished dataset creation')
        print('Number of sentences: %s' % len(tbc))

        inverse_vocab = {tbc.vocab[k]:k for k in tbc.vocab}

        print('Vocab size: %s' % len(tbc.vocab))

        messages = convert_numpy_array_to_strings(tbc.data[:1000, :], inverse_vocab,
                                                  tbc.stop_token,
                                                  keep_stop_token=False)

        for i in range(len(messages)):
            print(messages[i])
