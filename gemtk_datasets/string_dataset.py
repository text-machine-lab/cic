"""David Donahue November 2017"""
import gemtk
import gensim
import spacy
import numpy as np

class StringDataset(gemtk.Dataset):
    def __init__(self, strings, max_length, token_to_id=None, stop_token='<STOP>'):
        self.stop_token = stop_token
        self.max_message_length = max_length
        self.strings = strings
        print(max_length)
        self.nlp = spacy.load('en_core_web_sm')

        # Create or reuse vocabulary
        if token_to_id is None:
            self.token_to_id, self.id_to_token = create_vocabulary(self.strings)
            vocab_size = len(self.token_to_id)
            self.token_to_id[self.stop_token] = vocab_size
            self.id_to_token[vocab_size] = self.stop_token
        else:
            self.token_to_id = token_to_id
            self.id_to_token = {v: k for k, v in token_to_id.items()}

        self.np_messages, self.formatted_and_filtered_strings = self.convert_strings_to_numpy(self.strings)

            # = construct_numpy_from_messages(self.strings,
            #                                              self.token_to_id,
            #                                              self.max_message_length)

    def convert_numpy_to_strings(self, np_messages):
        messages = convert_numpy_array_to_strings(np_messages, self.id_to_token,
                                                  self.stop_token,
                                                  keep_stop_token=False)
        return messages

    def convert_strings_to_numpy(self, strings):
        tk_token_strings = []
        for each_string in strings:
            tk_string = self.nlp.tokenizer(each_string.lower())
            tk_tokens = [str(token) for token in tk_string if str(token) != ' ' and str(token) in self.token_to_id]
            tk_tokens += [self.stop_token]
            if len(tk_tokens) < self.max_message_length:
                tk_token_strings.append(tk_tokens)
        np_messages = construct_numpy_from_messages(tk_token_strings, self.token_to_id, self.max_message_length)
        formatted_and_filtered_strings = [' '.join(tk_token_string) for tk_token_string in tk_token_strings]
        return np_messages, formatted_and_filtered_strings

    def get_vocabulary(self):
        return self.token_to_id, self.id_to_token

    def get_stop_token(self):
        return self.stop_token

    def __getitem__(self, index):
        return {'message': self.np_messages[index]}

    def __len__(self):
        return len(self.np_messages)


def create_vocabulary(messages):
    message_tokens = [message.split() for message in messages]
    token_to_id = gensim.corpora.Dictionary(documents=message_tokens).token2id
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
    np_messages = np.zeros([m, max_length], dtype=int)
    for i in range(np_messages.shape[0]):
        message = messages[i]
        for j, each_token in enumerate(message):
            if j < max_length:
                np_messages[i, j] = vocab_dict[each_token]
    return np_messages


def convert_numpy_array_to_strings(np_examples, vocabulary, stop_token=None, keep_stop_token=False):
    """Converts a numpy array of indices into a list of strings.

    np_examples - m x n numpy array of ints, where m is the number of
    strings encoded by indices, and n is the max length of each string
    vocab_dict - where vocab_dict[index] gives a word in the vocabulary
    with that index

    Returns: a list of strings, where each string is constructed from
    indices in the array as they appear in the vocabulary."""
    assert stop_token is not None or not keep_stop_token
    m = np_examples.shape[0]
    n = np_examples.shape[1]
    examples = []
    for i in range(m):
        each_example = ''
        for j in range(n):
            word_index = np_examples[i][j]
            word = vocabulary[word_index]
            if stop_token is not None and word == stop_token:
                if keep_stop_token:
                    if j > 0:
                        each_example += ' '
                    each_example += stop_token
                break
            if j > 0 and word != '':
                each_example += ' '
            each_example += word
        examples.append(each_example)
    return examples