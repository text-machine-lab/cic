"""Train and evaluate a sequence-to-sequence model with attention on the Cornell Movie Dialogue dataset.
Model takes as input a history of previous utterances and produces the next utterance."""

from sacred import Experiment
from cic.datasets.cmd_history import CornellMovieHistoryDataset
from cic.datasets.cmd_one_turn import CornellMovieConversationDataset
from cic.datasets.text_dataset import convert_numpy_array_to_strings
from cic.models.seq_to_seq import Seq2Seq
import cic.paths
import os
import numpy as np


ex = Experiment('s2sa_cmd')

@ex.config
def config():
    max_s_len = 10
    max_c_len = 30
    vocab_len = 10000
    word_size = 200  # size of learned word embedding
    rnn_size = 200
    num_epochs = 30
    num_s_print = 100  # num val sentences to print
    lr = 0.0001  # learning rate
    keep_prob = 0.5
    cmd_save_dir = os.path.join(cic.paths.DATA_DIR, 'cornell_history_convos/')
    save_dir = os.path.join(cic.paths.DATA_DIR, 'cmd_s2sa/')
    attention=False

    gen_codes_and_save = True  # generate codes for all examples in dataset, save to disk
    save_codes_path = os.path.join(cic.paths.DATA_DIR, 'cmd_context_codes.npy')

    restore=False


def generate_codes_and_save_to_dir(dataset, model, save_path):
    """Use dataset to generate latent features from model. Save as .npy
    file to disk.

    dataset - instance of Dataset containing 'message' features
    model - instance of GenericModel mapping 'message'-->'code'
    save_path - path to save latent feature numpy array to"""

    codes = model.predict(dataset, outputs=['code'])

    np.save(save_path, codes)


@ex.automain
def main(max_s_len, max_c_len, vocab_len, word_size, rnn_size, num_epochs, num_s_print, restore, lr,
         keep_prob, save_dir, attention, gen_codes_and_save, save_codes_path):

    ds = CornellMovieHistoryDataset(max_vocab=vocab_len, max_c_len=max_c_len, max_s_len=max_s_len)

    # cornell_dir = os.path.join(cic.paths.DATA_DIR, 'cornell')
    #
    # ds = CornellMovieConversationDataset(max_s_len, reverse_inputs=False, seed='seed',
    #                                      save_dir=cornell_dir, max_vocab_len=vocab_len,
    #                                      regenerate=False)

    inv_vocab = ds.inv_vocab  # mapping from indices to words

    print('Vocab len: %s' % len(inv_vocab))

    ds_fm = ds.rename({'context': 'message'})  # compatibility with seq 2 seq interface (fm=formatted)

    train, val = ds_fm.split(0.99, seed='seed')

    print('Len train, val: %s, %s' % (len(train), len(val)))

    s2sa = Seq2Seq(max_c_len, max_s_len, len(ds.vocab), word_size, rnn_size, attention=attention, restore=restore,
                   save_dir=save_dir)

    s2sa.train(train, params={'learning rate': lr, 'keep_prob': keep_prob}, num_epochs=num_epochs)

    np_val_responses = s2sa.generate_responses(val, n=10)
    val_contexts = convert_numpy_array_to_strings(val.to_numpy('message'), inv_vocab, stop_token='<STOP>')
    val_responses = convert_numpy_array_to_strings(np_val_responses, inv_vocab, stop_token='<STOP>')

    for i in range(num_s_print):
        print(val_contexts[i])
        print('==> %s' % val_responses[i])
        print()

    if gen_codes_and_save:
        generate_codes_and_save_to_dir(ds_fm, s2sa, save_codes_path)






