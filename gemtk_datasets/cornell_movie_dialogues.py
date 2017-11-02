"""Stores Dataset subclass for the Cornell movie dialogues dataset."""
import numpy as np
import spacy
import unittest2

import config
from gemtk_datasets.string_dataset import StringDataset
from question_answering import squad_dataset_tools as sdt


class CornellMovieDialoguesDataset(StringDataset):
    def __init__(self, max_message_length=30, token_to_id=None, num_examples=None):
        """Currently creates a dataset of utterances from the dataset, for use in training an autoencoder.
        Does not return messsage -> response pairs as of now. max_message_length includes stop token, so really
        the largest sentence the autoencoder can encode is one less than the max length!"""
        self.num_examples = num_examples

        stop_token = '<STOP>'

        messages = self._load_messages_from_cornell_movie_lines(config.CORNELL_MOVIE_LINES_FILE,
                                                                     max_number_of_messages=self.num_examples)

        super().__init__(messages, max_message_length, token_to_id=token_to_id, stop_token=stop_token)

    def _load_messages_from_cornell_movie_lines(self, movie_lines_filename, max_number_of_messages=None):
        delimiter = ' +++$+++ '
        movie_lines_file = open(movie_lines_filename, 'rb')
        messages = []
        line_index = 0
        for message_line in movie_lines_file:
            if max_number_of_messages is None or line_index < max_number_of_messages:
                try:
                    message_data = message_line.decode('utf-8').split(delimiter)
                    message = message_data[4][:-1]
                    messages.append(message)
                except UnicodeDecodeError:
                    pass
            else:
                break

        movie_lines_file.close()

        return messages


class CornellMovieDialoguesTest(unittest2.TestCase):
    def test_empty(self):
        pass

    def test_reconstruct(self):
        """Confirm that all strings converted into numpy form can be converted back to their original string form.
        Excludes strings that go beyond max length limit (these would get cut off)."""
        cmd_dataset = CornellMovieDialoguesDataset()

        token_to_id, id_to_token = cmd_dataset.get_vocabulary()
        assert len(token_to_id) == len(id_to_token)
        nlp = spacy.load('en_core_web_sm')
        m = len(cmd_dataset)
        assert m == len(cmd_dataset.formatted_and_filtered_strings)
        for index in range(m):
            each_example = cmd_dataset[index]
            each_np_message = np.reshape(each_example['message'], newshape=(-1, 30))
            each_reconstructed_message = sdt.convert_numpy_array_to_strings(each_np_message, id_to_token,
                                                                            stop_token=cmd_dataset.stop_token,
                                                                            keep_stop_token=True)[0]
            each_message = cmd_dataset.formatted_and_filtered_strings[index]
            if len(cmd_dataset.formatted_and_filtered_strings[index].split()) <= 30:
                if each_message != each_reconstructed_message:
                    print(each_message)
                    print(each_reconstructed_message)
                    exit()

    def test_len(self):
        cmd_dataset = CornellMovieDialoguesDataset(max_message_length=10)

        for index in range(cmd_dataset.np_messages.shape[0]):
            assert cmd_dataset.np_messages[index, -1] == cmd_dataset.token_to_id[cmd_dataset.stop_token] or \
                   cmd_dataset.np_messages[index, -1] == cmd_dataset.token_to_id['']

