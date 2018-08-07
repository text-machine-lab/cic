"""Toronto Book Corpus implemented as a subclass of StringDataset."""
from cic.datasets.text_dataset import TextDataset
import cic.paths
import os
import h5py
import pickle
from arcadian.dataset import Dataset
import random

class TorontoBookCorpus(Dataset):
    def __init__(self, max_s_len, result_path, max_num_s=None, max_part_len=100000,
                 stop_token='<STOP>', regenerate=False, vocab=None, load_to_mem=True,
                 second_file_first=False, max_vocab_len=None, shuffle=True, **kwargs):
        """Book Corpus provided by Toronto University, with approx. 70 million sentences. Contains
        a single feature 'message' of numpy encoded sentences. Sentences are filtered for min and
        max lengths, and sentences containing non-alphabetical non-period characters are removed.
        If loading entire dataset be sure to set load_to_mem=False so as to keep the full dataset on disk."""

        self.stop_token = stop_token
        self.max_num_s = max_num_s

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # Dataset is contained in two .txt files, and is way too big to store on disk
        # We partition dataset into subsets, process each one using a string dataset,
        # and save all to an h5py file.
        strings = []

        data_path = os.path.join(result_path, 'data.hdf5')
        vocab_path = os.path.join(result_path, 'vocab.pkl')

        results_exist = (os.path.isfile(data_path) and os.path.isfile(vocab_path))

        if not results_exist or regenerate:

            # Open dataset file for storing numpy sentences
            self.dataset_file = h5py.File(data_path, 'w')

            # Count lines in file
            def file_len(fname):
                with open(fname) as f:
                    for i, l in enumerate(f):
                        pass
                return i + 1

            p1_num_lines = file_len(cic.paths.BOOK_CORPUS_P1)
            p2_num_lines = file_len(cic.paths.BOOK_CORPUS_P2)
            num_lines = p1_num_lines + p2_num_lines

            if max_s_len is not None:
                num_lines = min(num_lines, max_num_s)

            print('Number of sentences: %s' % num_lines)

            print('Max number of sentences: %s' % max_num_s)

            data = self.dataset_file.create_dataset('messages', (1, max_s_len), maxshape=(num_lines, max_s_len), dtype='i')

            assert 'messages' in self.dataset_file

            def readfiles(filenames):
                for f in filenames:
                    with open(f) as file:
                        for line in file:
                            yield line

            nlp = None
            num_read_s = 0  # number of sentences read and converted

            # This makes it easy to create a validation set using the second file.
            if second_file_first:
                files_to_read = [cic.paths.BOOK_CORPUS_P2, cic.paths.BOOK_CORPUS_P1]
            else:
                files_to_read = [cic.paths.BOOK_CORPUS_P1, cic.paths.BOOK_CORPUS_P2]

            for line in readfiles(files_to_read):

                line_tokens = line.split()
                line = ' '.join(line_tokens)

                accept = True

                # Any filtering of strings goes here.

                # Sentences can only contain letters and periods
                for c in line:
                    if not c.isalpha() and c != ' ' and c != '.':
                        accept = False

                # Remove sentences with duplicate adjacent words
                for index in range(len(line_tokens) - 1):
                    if line_tokens[index] == line_tokens[index+1]:
                        accept = False

                if accept:
                    strings.append(line)
                else:
                    continue

                # Quit when max_num_s strings are read
                if max_num_s is not None and num_read_s + len(strings) >= max_num_s:
                    print('break')
                    break

                if len(strings) >= max_part_len:
                    print('Number of examples previously read: %s' % num_read_s)

                    converter = TextDataset(strings, max_s_len, result_save_path=None, regenerate=True,
                                            token_to_id=vocab, update_vocab=True, stop_token=stop_token,
                                            nlp=nlp, max_vocab_len=max_vocab_len, **kwargs)

                    # Check that previous words still have the same indices
                    if vocab is not None:
                        for key in vocab:
                            assert vocab[key] == converter.token_to_id[key]

                    # Grab modified vocabulary, keep updating it
                    vocab = converter.token_to_id

                    # Grab tokenizer to save time
                    nlp = converter.nlp

                    # We converted one batch of strings, get ready for the next
                    strings = []

                    # Write to dataset
                    data.resize(num_read_s + len(converter), axis=0)
                    data[num_read_s:num_read_s+len(converter), :] = converter.np_messages


                    # Switch to next partition
                    num_read_s += len(converter)

            # Move remaining examples
            converter = TextDataset(strings, max_s_len, result_save_path=None, regenerate=True,
                                    token_to_id=vocab, update_vocab=True, stop_token=stop_token,
                                    nlp=nlp, **kwargs)

            # Check that previous words still have the same indices
            if vocab is not None:
                for key in vocab:
                    assert vocab[key] == converter.token_to_id[key]

            vocab = converter.token_to_id

            # Save vocab
            self.vocab = vocab
            with open(vocab_path, 'wb') as f:
                pickle.dump(vocab, f)

            data.resize(num_read_s + len(converter), axis=0)
            data[num_read_s:num_read_s + len(converter), :] = converter.np_messages

            assert data.shape[0] == num_read_s + len(converter)

            if shuffle:
                print('Shuffling data')
                random.shuffle(data)

            self.data = data

        else:
            print('Loading dataset from save...')

            self.dataset_file = h5py.File(data_path, 'r+')

            # for obj in self.dataset_file:
            #     print(obj)

            # Results exist, just load them!
            self.data = self.dataset_file['messages']

            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)

        # Get ready to index dataset
        if load_to_mem:
            self.data = self.data.value  # Retrieve numpy array from h5py array (disk to memory)

    def __getitem__(self, index):
        return {'message': self.data[index, :]}

    def __len__(self):
        if self.max_num_s is None:
            return self.data.shape[0]
        else:
            return min(self.data.shape[0], self.max_num_s)
