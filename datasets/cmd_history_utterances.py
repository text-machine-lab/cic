"""Cornell Movie Dialogues corpus, where conversations are represented each as a (history, response) pair,
where history are the N previous messages and response is how the systems responds."""
import cic.utils.mdd_tools as mddt
from arcadian.dataset import Dataset
from cic.utils.squad_tools import invert_dictionary
import cic.paths as paths
import spacy
import numpy as np

class CornellMovieHistoryUtteranceDataset(Dataset):

    def __init__(self, n=5, num_convos=None, max_vocab=10000, max_s_len=10, stop_token='<STOP>'):

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
        mddt.load_messages_from_cornell_movie_lines_by_id(id_to_msg, paths.CORNELL_MOVIE_LINES_FILE, stop_token,
                                                          self.nlp)

        none_count = [1 for id in id_to_msg if id_to_msg[id] is None]
        print('Fraction none: %s' % (np.sum(none_count) / len(id_to_msg)))

        convos = rm_convos_greater_max_len(convos, id_to_msg, max_s_len)

        #examples = self.build_examples_from_convos(convos, id_to_msg)

        print(convos[0])
        for index, key in enumerate(id_to_msg):
            print(key)
            print(id_to_msg[key])

            if index > 10:
                break

        # build vocabulary
        self.vocab = mddt.build_vocabulary_from_messages(id_to_msg, max_vocab_len=max_vocab)
        self.inv_vocab = invert_dictionary(self.vocab)

        self.np_convos = mddt.conversations_to_numpy(convos, id_to_msg, self.vocab, n, max_s_len,
                                                       add_stop=False)

    def build_examples_from_convos(self):
        pass

    def __len__(self):
        return self.np_convos.shape[0]

    def __getitem__(self, index):
        return {'convo': self.np_convos[index, :, :]}


def rm_convos_greater_max_len(convos, id2msg, max_len):
    """Removes conversations which contain messages that are
    greater than the max message length.

    convos - list of conversations from CMD dataset
    max_len - max message length
    id2msg - mapping from CMD message ids to actual message text

    Returns: a list of conversations that are below max len"""
    filt_convos = []

    for convo in convos:
        msg_ids = convo[3]
        msg_infos = [id2msg[msg_id] for msg_id in msg_ids]
        msgs = [msg_info[4] for msg_info in msg_infos if msg_info is not None]
        under_max_len=True # assume all messages under max length
        for msg in msgs:
            if len(msg) > max_len:
                # Detected message about max len. Don't add this convo
                under_max_len = False

        if under_max_len:
            filt_convos.append(convo)

    return filt_convos


# N = 5
# stop_token = '<STOP>'
# unk_token = '<UNK>'
#
# num_convos = None  # load all convos
# max_vocab_len = 10000000
# max_s_len = 20  # maximum sentence length
#
# ds = CornellMovieHistoryDataset(N, num_convos, max_vocab=max_vocab_len, max_s_len=max_s_len)
#
# print('Num convos: %s' % len(ds))
#
# for i in range(len(ds)):
#     if i >= 10:
#         break
#
#     for j in range(ds.np_convos.shape[1]):
#
#         msg_tokens = []
#         for k in range(ds.np_convos.shape[2]):
#             msg_tokens.append(ds.inv_vocab[ds.np_convos[i, j, k]])
#
#         print(' '.join(msg_tokens))
#
#     print()
















