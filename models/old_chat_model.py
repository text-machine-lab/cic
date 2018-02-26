"""Supporting functions used in chat_model.py"""
import numpy as np
import random
import unittest2
import pickle
import os
from cic.utils import squad_tools as sdt, mdd_tools as mddt

from cic import config

LEARNING_RATE = .0008
NUM_CONVERSATIONS = None
NUM_EXAMPLES_TO_PRINT = 20
MAX_MESSAGE_LENGTH = 10
LEARNED_EMBEDDING_SIZE = 100
KEEP_PROB = 0.5
RNN_HIDDEN_DIM = 1000
TRAIN_FRACTION = 0.9
BATCH_SIZE = 20
NUM_EPOCHS = 200
RESTORE_FROM_SAVE = False
REVERSE_INPUT_MESSAGE = True
SHUFFLE_EXAMPLES = True
STOP_TOKEN = '<STOP>'
SEQ2SEQ_IMPLEMENTATION = 'homemade'  # 'homemade', 'dynamic_rnn', 'keras'

class BatchGenerator:
    def __init__(self, datas, batch_size):
        if isinstance(datas, list):
            self.datas = datas
        else:
            self.datas = [datas]
        self.batch_size = batch_size

    def generate_batches(self):
        m = self.datas[0].shape[0]
        num_batches = int(m / self.batch_size + 1)
        for batch_index in range(num_batches):
            if batch_index == num_batches - 1:
                real_batch_size = m - batch_index * self.batch_size
            else:
                real_batch_size = self.batch_size

            if real_batch_size == 0:
                break

            batch = [np_data[self.batch_size*batch_index:self.batch_size*batch_index+real_batch_size] for np_data in self.datas]
            if len(batch) > 1:
                yield batch
            else:
                yield batch[0]


def preprocess_all_cornell_conversations(nlp, vocab_dict=None, reverse_inputs=True, verbose=True,
                                         keep_duplicates=False, seed='hello world', stop_token='<STOP>',
                                         max_message_length=MAX_MESSAGE_LENGTH, save_dir=None,
                                         max_vocab_len=None, regen=False):
    """All preprocessing of conversational data for run_old_chat_model.py. This function is also
    intended to be used by run_latent_chat.py"""
    # seed so that train and validation examples don't get blended together.
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if regen or save_dir is None or not os.path.isfile(os.path.join(save_dir, 'cornell_convos.pkl')):
        random.seed(seed)
        if verbose:
            print('Processing conversations...')
        conversations, id_to_message = mddt.load_cornell_movie_dialogues_dataset(config.CORNELL_MOVIE_CONVERSATIONS_FILE,
                                                                                 max_conversations_to_load=NUM_CONVERSATIONS)
        if verbose:
            print('Number of valid conversations: %s' % len(conversations))

            print('Finding messages...')
        mddt.load_messages_from_cornell_movie_lines_by_id(id_to_message, config.CORNELL_MOVIE_LINES_FILE, stop_token, nlp)

        num_messages = len(id_to_message)
        if verbose:
            print('Number of messages: %s' % num_messages)

        num_empty_messages = 0
        message_lengths = []
        for key in id_to_message:
            if id_to_message[key] is None:
                num_empty_messages += 1
            else:
                message_lengths.append(len(id_to_message[key][-1]))
        np_message_lengths = np.array(message_lengths)
        if verbose:
            print('Number of missing messages: %s' % num_empty_messages)
            print('Average message length: %s' % np.mean(np_message_lengths))
            print('Message length std: %s' % np.std(np_message_lengths))
            print('Message max length: %s' % np.max(np_message_lengths))

        if vocab_dict is None:
            vocab_dict = mddt.build_vocabulary_from_messages(id_to_message, max_vocab_len=max_vocab_len)
        vocabulary = sdt.invert_dictionary(vocab_dict)
        vocabulary_length = len(vocab_dict)
        if verbose:
            print('Vocabulary size: %s' % vocabulary_length)

        examples = mddt.construct_examples_from_conversations_and_messages(conversations, id_to_message,
                                                                           max_message_length=max_message_length)

        if verbose:
            print('Creating examples...')
        num_examples = len(examples)

        if verbose:
            print('Example example: %s' % str(examples[0]))
            print('Number of examples: %s' % num_examples)

        non_duplicate_messages = []
        non_duplicate_responses = []
        num_duplicates = 0
        for each_example in examples:
            each_message = each_example[0]
            each_response = each_example[1]
            if each_message not in non_duplicate_messages:
                non_duplicate_messages.append(each_message)
                non_duplicate_responses.append(each_response)
            else:
                num_duplicates += 1
        print('Number of duplicate examples: %s' % num_duplicates)

        non_duplicate_examples = [(non_duplicate_message, non_duplicate_response)
                                  for non_duplicate_message, non_duplicate_response
                                  in zip(non_duplicate_messages, non_duplicate_responses)]

        if not keep_duplicates:
            print('Removing duplicate examples')
            examples = non_duplicate_examples
            print('Number of remaining examples: %s' % len(examples))

        if SHUFFLE_EXAMPLES:
            random.shuffle(examples)

        if verbose:
            print('Constructing input numpy arrays...')
        np_message, np_response = construct_numpy_from_examples(examples, vocab_dict, max_message_length)
        # if verbose:
        #     print('Validating inputs...')
        # message_reconstruct = sdt.convert_numpy_array_to_strings(np_message, vocabulary)
        # response_reconstruct = sdt.convert_numpy_array_to_strings(np_response, vocabulary)
        # for i in range(len(examples)):
        #     each_message = ' '.join(examples[i][0])
        #     each_response = ' '.join(examples[i][1])
        #     # print(message_reconstruct[i])
        #     # print(response_reconstruct[i])
        #
        #     if len(examples[i][0]) <= max_message_length:
        #         assert each_message == message_reconstruct[i]
        #     if len(examples[i][1]) <= max_message_length:
        #         assert each_response == response_reconstruct[i]

        if reverse_inputs:
            if verbose:
                print('Reversing input arrays (improves performance)')
            np_message = np.flip(np_message, axis=1)

        results = [examples, np_message, np_response, vocab_dict, vocabulary]

        if save_dir is not None:
            pickle.dump(results, open(os.path.join(save_dir, 'cornell_convos.pkl'), 'wb'))
    else:
        results = pickle.load(open(os.path.join(save_dir, 'cornell_convos.pkl'), 'rb'))

    return results


def construct_numpy_from_examples(examples, vocab_dict, max_length):
    """Convert Movie corpus message pairs into numpy arrays by token index
    in vocab_dict."""
    first_messages = [example[0] for example in examples]
    second_messages = [example[1] for example in examples]
    np_first = construct_numpy_from_messages(first_messages, vocab_dict, max_length)
    np_second = construct_numpy_from_messages(second_messages, vocab_dict, max_length)
    return np_first, np_second


def construct_numpy_from_messages(messages, vocab_dict, max_length):
    """Construct a numpy array from messages using vocab_dict as a mapping
    from each word to an integer index. If out-of-vocabulary word, it
    uses the <UNK> token!"""
    m = len(messages)
    np_messages = np.zeros([m, max_length])
    for i in range(np_messages.shape[0]):
        message = messages[i]
        for j, each_token in enumerate(message):
            if j < max_length:
                if each_token in vocab_dict:
                    np_messages[i, j] = vocab_dict[each_token]
                else:
                    np_messages[i, j] = vocab_dict['<UNK>']
    return np_messages


class ChatModelFuncTest(unittest2.TestCase):
    def setUp(self):
        pass

    def test_construct_numpy_from_messages(self):
        message_one = ['i', 'like', 'walking', 'to', 'the', 'park']
        message_two = ['we', 'like', 'walking', 'on', 'the', 'beach']
        messages = [message_one, message_two]
        print(messages)
        vocab_dict = {'': 0, 'i': 1, 'like': 2, 'walking': 3, 'to': 4, 'the': 5, 'park': 6,
                      'we': 7, 'on': 8, 'beach': 9}
        np_messages = construct_numpy_from_messages(messages, vocab_dict, 7)
        assert np.array_equal(np_messages, np.array([[1, 2, 3, 4, 5, 6, 0], [7, 2, 3, 8, 5, 9, 0]]))

        examples = [[message_one, message_two]]
        np_first, np_second = construct_numpy_from_examples(examples, vocab_dict, 7)
        print(np_first)
        print(np_second)
        print(np_messages)
        assert np.array_equal(np_first[0, :], np_messages[0, :])
        assert np.array_equal(np_second[0, :], np_messages[1, :])

    def test_batch_generator(self):
        np_values = np.random.uniform(size=(10, 5))
        gen = BatchGenerator(np_values, 20)
        all_batches = []
        for np_batch in gen.generate_batches():
            all_batches.append(np_batch)
        np_collected_values = np.concatenate(all_batches, axis=0)
        assert np.array_equal(np_values, np_collected_values)
