"""Toronto Book Corpus implemented as a subclass of StringDataset."""
from cic.gemtk_datasets.string_dataset import StringDataset
import cic.config
import os
import h5py
import pickle
from arcadian.dataset import Dataset

class TorontoBookCorpus(Dataset):
    def __init__(self, max_s_len, result_path, max_num_s=None, max_part_len=100000,
                 stop_token='<STOP>', regenerate=False, vocab=None, **kwargs):
        """Book Corpus provided by Toronto University, with approx. 70 million sentences. Yay!"""

        self.stop_token = stop_token

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # Dataset is contained in two .txt files, and is way too big to store on disk
        # We partition dataset into a series of StringDatasets and save them all to disk separately
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

            p1_num_lines = file_len(cic.config.BOOK_CORPUS_P1)
            p2_num_lines = file_len(cic.config.BOOK_CORPUS_P2)
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

            for line in readfiles([cic.config.BOOK_CORPUS_P1, cic.config.BOOK_CORPUS_P2]):

                line_tokens = line.split()
                line = ' '.join(line_tokens)

                accept = True

                # Any filtering of strings goes here.
                for c in line:
                    if c.isdigit():
                        accept = False

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

                    converter = StringDataset(strings, max_s_len, result_save_path=None, regenerate=True,
                                              token_to_id=vocab, update_vocab=True, stop_token=stop_token,
                                              nlp=nlp, **kwargs)

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
            converter = StringDataset(strings, max_s_len, result_save_path=None, regenerate=True,
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

            # Get ready to index dataset
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

    def __getitem__(self, index):
        return {'message': self.data[index, :]}

    def __len__(self):
        return self.data.shape[0]
