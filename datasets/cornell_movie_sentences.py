"""Stores Dataset subclass for the Cornell movie dialogues dataset."""

from cic.datasets.text_dataset import TextDataset


class CornellMovieSentencesDataset(TextDataset):
    def __init__(self, cornell_movie_lines_file, max_s_len=30, token_to_id=None, num_examples=None, regenerate=False):
        """A flat array of strings from the Cornell Movie Dialogues dataset. Does not return message-response pairs
        as of now. Data contains one feature 'message' which is a numpy encoded string."""
        self.num_examples = num_examples

        stop_token = '<STOP>'

        messages = self._load_messages_from_cornell_movie_lines(cornell_movie_lines_file,
                                                                     max_number_of_messages=self.num_examples)

        super().__init__(messages, max_s_len, token_to_id=token_to_id, stop_token=stop_token, regenerate=regenerate)

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

