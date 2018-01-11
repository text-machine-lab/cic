"""Train and evaluate neural language model on Toronto Book Corpus."""

from cic.datasets.book_corpus import TorontoBookCorpus
from cic.datasets.text_dataset import convert_numpy_array_to_strings
from cic.models.nlm import NeuralLanguageModelTraining, NeuralLanguageModelPrediction
import cic.config
from sacred import Experiment
import numpy as np
import os
import pickle

ex = Experiment('nlm')

@ex.config
def config():
    max_num_s = 2000000  # number of sentences to train on
    max_len = 20
    emb_size = 200  # word embedding size
    rnn_size = 600  # lstm hidden state size
    restore = False  # restore model from saved parameters
    save_dir = cic.config.NLM_SAVE_DIR  # where to store model parameters
    num_epochs = 5
    num_samples = 2000  # number of sentences to generate after training


@ex.automain
def main(max_num_s, max_len, emb_size, rnn_size, restore, save_dir, num_epochs, num_samples):

    ds = TorontoBookCorpus(20, result_path=cic.config.BOOK_CORPUS_RESULT,
                            min_length=5, max_num_s=max_num_s, keep_unk_sentences=False,
                            vocab_min_freq=5, vocab=None, regenerate=False)

    nlm_train = NeuralLanguageModelTraining(max_len, len(ds.vocab), emb_size, rnn_size, save_dir=save_dir,
                                            tensorboard_name='nlm', restore_from_save=restore)

    if num_epochs > 0:
        nlm_train.train(ds, num_epochs=num_epochs)

    print('Generating sentences')

    nlm_predict = NeuralLanguageModelPrediction(len(ds.vocab), emb_size, rnn_size, save_dir=save_dir,
                                                tensorboard_name='nlm', restore_from_save=True)

    go_token = nlm_predict.predict(None, output_tensor_names=['go_token'])['go_token']

    init_hidden = nlm_predict.predict(None, output_tensor_names=['init_hidden'])['init_hidden']

    prev_word_embs = np.repeat(go_token, num_samples, axis=0)
    hidden_s = np.repeat(init_hidden, num_samples, axis=0)

    sampled_sentence_words = []
    for t in range(max_len):
        result = nlm_predict.predict({'hidden': hidden_s, 'teacher_signal': prev_word_embs},
                                     output_tensor_names=['probabilities', 'hidden'])
        word_probs = result['probabilities']
        hidden_s = result['hidden']

        # sample word from probability distribution
        vocab_len = len(ds.vocab)

        np_words = np.zeros([num_samples])
        for ex_index in range(word_probs.shape[0]):  # example index
            word_index = np.random.choice(np.arange(vocab_len), p=word_probs[ex_index, :])
            np_words[ex_index] = word_index

        # grab embedding per word
        np_word_embs = nlm_predict.predict({'word': np_words}, output_tensor_names=['word_emb'])['word_emb']

        # set as next teacher signal and save word index
        prev_word_embs = np_word_embs
        sampled_sentence_words.append(np_words)

    np_messages = np.stack(sampled_sentence_words, axis=1)

    reversed_vocab = {ds.vocab[k]:k for k in ds.vocab}

    messages = convert_numpy_array_to_strings(np_messages, reversed_vocab,
                                            ds.stop_token,
                                            keep_stop_token=False)

    print()
    for message in messages:
        print(message)

    pickle.dump(messages, open(os.path.join(cic.config.DATA_DIR, 'nlm_messages.pkl'), 'wb'))