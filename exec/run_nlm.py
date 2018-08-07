"""Train and evaluate neural language model on Toronto Book Corpus."""

from cic.datasets.book_corpus import TorontoBookCorpus
from cic.datasets.text_dataset import convert_numpy_array_to_strings
from cic.models.nlm import NeuralLanguageModelTraining, NeuralLanguageModelPrediction
import cic.paths
from sacred import Experiment
import numpy as np
import os
import pickle

ex = Experiment('nlm')

@ex.config
def config():
    max_num_s = 16000000  # number of sentences to train on
    max_len = 20
    emb_size = 200  # word embedding size
    rnn_size = 600  # lstm hidden state size
    restore = False  # restore model from saved parameters
    save_dir = cic.paths.NLM_SAVE_DIR  # where to store model parameters
    num_epochs = 100
    num_samples = 2000  # number of sentences to generate after training
    max_vocab_len = 100000
    regen = False
    result_save_dir = os.path.join(cic.paths.DATA_DIR, 'nlm_messages.pkl')  # save generated sentences to disk
    embs_save_dir = os.path.join(cic.paths.DATA_DIR, 'nlm_embs.npy')  # save learned nlm embeddings to disk after prediction
    book_corpus_path = os.path.join(cic.paths.DATA_DIR, 'full_book_corpus/')
    shuffle = True  # shuffle during training
    load_to_mem = False  # load book corpus dataset entirely into memory (don't do for large max_num_s)

@ex.automain
def main(max_num_s, max_len, emb_size, rnn_size, restore, save_dir, regen, shuffle, load_to_mem,
         num_epochs, num_samples, embs_save_dir, result_save_dir, book_corpus_path, max_vocab_len):

    ds = TorontoBookCorpus(20, result_path=book_corpus_path,
                           min_length=5, max_num_s=max_num_s, keep_unk_sentences=False,
                           vocab_min_freq=5, vocab=None, regenerate=regen,
                           max_part_len=1000000, max_vocab_len=max_vocab_len,
                           load_to_mem=load_to_mem, shuffle=False)

    print('Num examples: %s' % len(ds))
    print('Vocab len: %s' % len(ds.vocab))

    nlm_train = NeuralLanguageModelTraining(max_len, len(ds.vocab), emb_size, rnn_size, save_dir=save_dir,
                                            tensorboard_name='nlm', restore=restore)

    nlm_train.shuffle = shuffle

    if num_epochs > 0:
        nlm_train.train(ds, num_epochs=num_epochs)

    print('Generating sentences')

    nlm_predict = NeuralLanguageModelPrediction(len(ds.vocab), emb_size, rnn_size, save_dir=save_dir,
                                                tensorboard_name='nlm', restore=True)

    go_token = nlm_predict.predict(None, outputs=['go_token'])

    init_hidden = nlm_predict.predict(None, outputs=['init_hidden'])

    prev_word_embs = np.repeat(go_token, num_samples, axis=0)
    hidden_s = np.repeat(init_hidden, num_samples, axis=0)

    sampled_sentence_words = []
    for t in range(max_len):
        result = nlm_predict.predict({'hidden': hidden_s, 'teacher_signal': prev_word_embs},
                                     outputs=['probabilities', 'hidden'])
        word_probs = result['probabilities']
        hidden_s = result['hidden']

        # sample word from probability distribution
        vocab_len = len(ds.vocab)

        np_words = np.zeros([num_samples])
        for ex_index in range(word_probs.shape[0]):  # example index
            word_index = np.random.choice(np.arange(vocab_len), p=word_probs[ex_index, :])
            np_words[ex_index] = word_index

        # grab embedding per word
        np_word_embs = nlm_predict.predict({'word': np_words}, outputs=['word_emb'])

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

    # Save generated responses to disk
    pickle.dump(messages, open(result_save_dir, 'wb'))

    # Save learned word embeddings to disk
    if embs_save_dir is not None:
        embs = nlm_predict.predict(None, outputs=['embs'])
        np.save(embs_save_dir, embs)

