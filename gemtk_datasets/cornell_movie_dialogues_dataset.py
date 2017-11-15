"""Stores Dataset subclass for the Cornell movie dialogues dataset."""
import numpy as np
import spacy
import unittest2

import config
from gemtk_datasets.string_dataset import StringDataset
from question_answering import squad_dataset_tools as sdt


class CornellMovieDialoguesDataset(StringDataset):
    def __init__(self, cornell_movie_lines_file, max_message_length=30, token_to_id=None, num_examples=None):
        """Currently creates a dataset of utterances from the dataset, for use in training an autoencoder.
        Does not return messsage -> response pairs as of now. max_message_length includes stop token, so really
        the largest sentence the autoencoder can encode is one less than the max length!"""
        self.num_examples = num_examples

        stop_token = '<STOP>'

        messages = self._load_messages_from_cornell_movie_lines(cornell_movie_lines_file,
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

