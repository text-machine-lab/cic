"""Cornell Movie Dialogues corpus, where conversations are represented each as a (history, response) pair,
where history are the N previous messages and response is how the systems responds."""
import cic.utils.mdd_tools as mddt
from arcadian.dataset import Dataset
from cic.utils.squad_tools import invert_dictionary
from cic.datasets.text_dataset import construct_numpy_from_messages, convert_numpy_array_to_strings
import cic.paths as paths
import pickle
import spacy
import numpy as np
import os

class CornellMovieHistoryDataset(Dataset):

    def __init__(self, num_convos=None, max_vocab=10000, max_c_len=50, max_s_len=10, save_dir=None, regen=False):
        """Creates a dataset of (context, target) pairs from the Cornell Movie Dialogue dataset, where context
        is the previous utterances in the conversation up until the current turn, and target is the next
        utterance to be spoken. The context feature is of shape (num_utterances, max_c_len) where num_utterances
        is the size of the dataset (every utterance can be used to create an example. The target feature is of
        shape (num_utterances, max_s_len).

        num_convos - max number of conversations (default None to use all convos)
        max_vocab - maximum size of vocabulary (<UNK> is used for out-of-vocab words
        max_c_len - maximum number of tokens in context
        max_s_len - maximum number of tokens in target utterance
        stop_token - string to use as stop token in vocab
        save_dir - save intermediate results to this directory for faster loading
        regen - regenerate intermediate results (does by default if save_dir=None)

        """
        self.stop_token = '<STOP>'

        if save_dir is None or regen:

            convos, id_to_msg = mddt.load_cornell_movie_dialogues_dataset(paths.CORNELL_MOVIE_CONVERSATIONS_FILE,
                                                                          max_conversations_to_load=num_convos)

            self.nlp = spacy.load('en')

            print('Number of valid conversations: %s' % len(convos))

            convo_lens = [len(conversation[3]) for conversation in convos]
            print('Avg convo len: %s' % np.mean(convo_lens))
            print('Std convo len: %s' % np.std(convo_lens))
            print('Min convo len: %s' % np.min(convo_lens))
            print('Max convo len: %s' % np.max(convo_lens))
            print('Finding messages...')
            mddt.load_messages_from_cornell_movie_lines_by_id(id_to_msg, paths.CORNELL_MOVIE_LINES_FILE, '<STOP>',
                                                              self.nlp)

            none_count = [1 for id in id_to_msg if id_to_msg[id] is None]
            print('Fraction none: %s' % (np.sum(none_count) / len(id_to_msg)))

            #convos = rm_convos_greater_max_len(convos, id_to_msg, max_s_len)
            # convert conversations to format of list of lists of messages
            convos = format_convos(convos, id_to_msg)

            examples = build_examples_from_convos(convos, max_c_len=max_c_len, max_s_len=max_s_len)
            contexts = [example[0] for example in examples]
            targets = [example[1] for example in examples]

            # build vocabulary
            self.vocab = mddt.build_vocabulary_from_messages(id_to_msg, max_vocab_len=max_vocab)
            self.inv_vocab = invert_dictionary(self.vocab)

            self.np_contexts = construct_numpy_from_messages(contexts, self.vocab, max_c_len, unk_token='<UNK>')
            self.np_targets = construct_numpy_from_messages(targets, self.vocab, max_s_len, unk_token='<UNK>')

            # save intermediate results
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
                    pickle.dump([self.np_contexts, self.np_targets, self.vocab, self.inv_vocab], f)
        else:
            # load intermediate results
            with open(os.path.join(save_dir, 'results.pkl'), 'rb') as f:
                self.np_contexts, self.np_targets, self.vocab, self.inv_vocab = pickle.load(f)

    def __len__(self):
        return self.np_targets.shape[0]

    def __getitem__(self, index):
        return {'context': self.np_contexts[index, :],
                'response': self.np_targets[index, :]}


def build_examples_from_convos(convos, max_c_len=None, max_s_len=None):
    """Turns each conversation into examples,
    for each utterance in the conversation. Removes
    examples which have contexts or responses that
    are greater than the provided max lengths.

    convos - list of lists of messages (list of conversations)
    max_c_len - maximum length of each context
    max_s_len - maximum length of response

    Returns: returns examples of form (prev_utterances, utterance)."""
    examples = []
    for convo in convos:
        for index in range(len(convo)):
            response = convo[index]
            prev_msgs = convo[:index]
            context = sum(prev_msgs, [])

            if max_c_len is None or len(context) <= max_c_len:
                if max_s_len is None or len(response) <= max_s_len:
                    examples.append([context, response])

    return examples


def format_convos(convos, id2msg):
    """Convert the format of conversations from the dataset into
    the form of a list of conversations (or a list of lists of messages).

    Returns: a list of lists of strings."""
    formatted_convos = []
    for convo in convos:
        msg_ids = convo[3]
        msg_infos = [id2msg[msg_id] for msg_id in msg_ids]
        msgs = [msg_info[4] for msg_info in msg_infos if msg_info is not None]
        formatted_convos.append(msgs)
    return formatted_convos


# test code
if __name__ == '__main__':
    ds = CornellMovieHistoryDataset(num_convos=None, max_vocab=10000, max_c_len=50, max_s_len=10)

    print('Num convos: %s' % len(ds))

    contexts = convert_numpy_array_to_strings(ds.np_contexts, ds.inv_vocab)
    targets = convert_numpy_array_to_strings(ds.np_targets, ds.inv_vocab)

    for i in range(100):
        print('%s | %s' % (contexts[i], targets[i]))
















