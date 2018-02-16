"""Train and evaluate chat model trained on Cornell Movie Dialogues."""
import sacred
from cic.datasets.cornell_movie_conversation import CornellMovieConversationDataset
from cic.datasets.text_dataset import convert_numpy_array_to_strings
from cic.models.seq_to_seq import Seq2Seq
import cic.config
import os
import numpy as np

ex = sacred.Experiment('chat')

@ex.config
def config():
    max_s_len = 10
    emb_size = 200
    rnn_size = 200
    num_epochs = 10
    restore=False

    split_frac = 0.95
    split_seed = 'seed'

    save_dir = os.path.join(cic.config.DATA_DIR, 'chat_model/')
    cornell_dir = os.path.join(cic.config.DATA_DIR, 'cornell_convos/')

@ex.automain
def main(max_s_len, emb_size, rnn_size, num_epochs, split_frac, split_seed, save_dir, restore, cornell_dir):
    print('Starting program')

    ds = CornellMovieConversationDataset(max_s_len, reverse_inputs=False, seed='seed', save_dir=cornell_dir)

    train_ds, val_ds = ds.split(split_frac, seed=split_seed)

    model = Seq2Seq(max_s_len, len(ds.vocab), emb_size, rnn_size,
                    save_dir=save_dir, restore=restore, tensorboard_name='chat')

    if num_epochs > 0:
        model.train(train_ds, num_epochs=num_epochs)

    go_token = model.predict(None, outputs=['go_token'])

    init_hidden = np.zeros([1, rnn_size * 2])

    num_samples = len(val_ds)

    prev_word_embs = np.repeat(go_token, num_samples, axis=0)
    hidden_s = np.repeat(init_hidden, num_samples, axis=0)

    print('Shape hidden_s: %s' % str(hidden_s.shape))

    codes = model.predict(val_ds, outputs=['codes'])

    sampled_sentence_words = []
    for t in range(max_s_len):
        result = model.predict([val_ds, {'state': hidden_s, 'input_word_emb': prev_word_embs, 'code': codes}],
                                     outputs=['word_prob', 'word_state'])

        word_probs = result['word_prob']
        hidden_s = result['word_state']

        # sample word from probability distribution
        vocab_len = len(ds.vocab)

        np_words = np.zeros([num_samples])
        for ex_index in range(word_probs.shape[0]):  # example index
            word_index = np.random.choice(np.arange(vocab_len), p=word_probs[ex_index, :])
            np_words[ex_index] = word_index

        # grab embedding per word
        np_word_embs = model.predict({'word': np_words}, outputs=['word_emb'])

        # set as next teacher signal and save word index
        prev_word_embs = np_word_embs
        sampled_sentence_words.append(np_words)

    np_messages = np.stack(sampled_sentence_words, axis=1)

    reversed_vocab = {ds.vocab[k]:k for k in ds.vocab}

    messages = convert_numpy_array_to_strings(np_messages, reversed_vocab,
                                            ds.stop_token,
                                            keep_stop_token=False)

    print()
    for index, message in enumerate(messages):
        print(' '.join(ds.examples[val_ds.indices[index]][0])),
        print(' --> ' + message)