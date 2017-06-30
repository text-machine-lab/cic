"""Supporting functions used in chat_model.py"""
import unittest2
import numpy as np

class BatchGenerator:
    def __init__(self, np_data, batch_size):
        self.np_data = np_data
        self.batch_size = batch_size

    def generate_batches(self):
        m = self.np_data.shape[0]
        num_batches = int(m / self.batch_size + 1)
        for batch_index in range(num_batches):
            if batch_index == num_batches - 1:
                real_batch_size = m - batch_index * self.batch_size
            else:
                real_batch_size = self.batch_size
            np_batch = self.np_data[real_batch_size*batch_index:real_batch_size*batch_index+real_batch_size]
            yield np_batch


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
    from each word to an integer index."""
    m = len(messages)
    np_messages = np.zeros([m, max_length])
    for i in range(np_messages.shape[0]):
        message = messages[i]
        for j, each_token in enumerate(message):
            if j < max_length:
                np_messages[i, j] = vocab_dict[each_token]
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
