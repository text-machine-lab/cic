"""Stores Dataset subclass for the Cornell movie dialogues dataset."""
import config
import gmtk
import spacy
import gensim
import numpy as np
import unittest2
import squad_dataset_tools as sdt


class CornellMovieDialoguesDataset(gmtk.Dataset):
    def __init__(self, max_message_length=30, token_to_id=None, num_examples=None):
        """Currently creates a dataset of utterances from the dataset, for use in training an autoencoder.
        Does not return messsage -> response pairs as of now. max_message_length includes stop token, so really
        the largest sentence the autoencoder can encode is one less than the max length!"""
        self.stop_token = '<STOP>'
        self.num_examples = num_examples
        self.max_message_length = max_message_length
        self.nlp = spacy.load('en_core_web_sm')
        # Add -1 to message length to include stop token
        self.messages = self._load_messages_from_cornell_movie_lines(config.CORNELL_MOVIE_LINES_FILE, self.nlp,
                                                                     max_number_of_messages=self.num_examples,
                                                                     max_message_length=self.max_message_length - 1,
                                                                     stop_token=self.stop_token)
        # Create or reuse vocabulary
        if token_to_id is None:
            self.token_to_id, self.id_to_token = create_vocabulary(self.messages)
        else:
            self.token_to_id = token_to_id
            self.id_to_token = {v: k for k, v in token_to_id.items()}

        self.np_messages = construct_numpy_from_messages(self.messages,
                                                         self.token_to_id,
                                                         self.max_message_length)

    def _load_messages_from_cornell_movie_lines(self, movie_lines_filename, nlp, max_number_of_messages=None,
                                               max_message_length=None, stop_token=None):
        delimiter = ' +++$+++ '
        movie_lines_file = open(movie_lines_filename, 'rb')
        messages = []
        line_index = 0
        for message_line in movie_lines_file:
            if max_number_of_messages is None or line_index < max_number_of_messages:
                try:
                    message_data = message_line.decode('utf-8').split(delimiter)
                    message = message_data[4][:-1]
                    tk_message = nlp.tokenizer(message.lower())
                    tk_tokens = [str(token) for token in tk_message if str(token) != ' ']
                    if stop_token is not None:
                        tk_tokens += [stop_token]
                    if max_message_length is None or len(tk_tokens) <= max_message_length:
                        messages.append(tk_tokens)
                        line_index += 1
                except UnicodeDecodeError:
                    pass
            else:
                break
        return messages

    def convert_numpy_to_strings(self, np_messages):
        messages = sdt.convert_numpy_array_to_strings(np_messages, self.id_to_token,
                                                                 self.stop_token,
                                                                 keep_stop_token=False)
        return messages

    def convert_strings_to_numpy(self, strings):
        tk_token_strings = []
        for each_string in strings:
            tk_string = self.nlp.tokenizer(each_string.lower())
            tk_tokens = [str(token) for token in tk_string if str(token) != ' ' and str(token) in self.token_to_id]
            tk_tokens += [self.stop_token]
            tk_token_strings.append(tk_tokens)

        return construct_numpy_from_messages(tk_token_strings, self.token_to_id, self.max_message_length)

    def get_vocabulary(self):
        return self.token_to_id, self.id_to_token

    def get_stop_token(self):
        return self.stop_token

    def __getitem__(self, index):
        return {'message': self.np_messages[index]}

    def __len__(self):
        return len(self.messages)


def create_vocabulary(messages):
    token_to_id = gensim.corpora.Dictionary(documents=messages).token2id
    id_to_token = {v: k for k, v in token_to_id.items()}
    # Add '' as index 0
    num_non_empty_words = len(token_to_id)
    token_to_id[id_to_token[0]] = num_non_empty_words
    token_to_id[''] = 0
    id_to_token[num_non_empty_words] = id_to_token[0]
    id_to_token[0] = ''

    return token_to_id, id_to_token


def construct_numpy_from_messages(messages, vocab_dict, max_length):
    """Construct a numpy array from messages using vocab_dict as a mapping
    from each word to an integer index."""
    m = len(messages)
    np_messages = np.zeros([m, max_length])
    for i in range(np_messages.shape[0]):
        message = messages[i]
        for j, each_token in enumerate(message):
            if j < max_length:
                np_messages[i, j] = vocab_dict[each_token]
    return np_messages


class CornellMovieDialoguesTest(unittest2.TestCase):
    def test_empty(self):
        pass

    def test_reconstruct(self):
        """Confirm that all strings converted into numpy form can be converted back to their original string form.
        Excludes strings that go beyond max length limit (these would get cut off)."""
        cmd_dataset = CornellMovieDialoguesDataset()

        token_to_id, id_to_token = cmd_dataset.get_vocabulary()

        for index in range(len(cmd_dataset)):
            each_example = cmd_dataset[index]
            each_np_message = np.reshape(each_example['message'], newshape=(-1, 30))
            each_reconstructed_message = sdt.convert_numpy_array_to_strings(each_np_message, id_to_token,
                                                                            stop_token=cmd_dataset.stop_token,
                                                                            keep_stop_token=True)[0]
            each_message = ' '.join(cmd_dataset.messages[index])
            if len(cmd_dataset.messages[index]) <= 30:
                if each_message != each_reconstructed_message:
                    print(each_message)
                    print(each_reconstructed_message)
                    exit()

    def test_len(self):
        cmd_dataset = CornellMovieDialoguesDataset(max_message_length=10)

        for each_message in cmd_dataset.messages:
            assert len(each_message) <= 10

