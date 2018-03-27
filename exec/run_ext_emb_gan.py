"""Train and evaluate the external word embedding GAN for text, trained using word embeddings from a pretrained model."""
from sacred import Experiment
from arcadian.dataset import MergeDataset
from cic.datasets.book_corpus import TorontoBookCorpus
from cic.datasets.text_dataset import convert_numpy_array_to_strings
from cic.models.rnet_gan import GaussianRandomDataset
from cic.models.ext_emb_gan import ExtEmbGAN
import cic.paths
import numpy as np
import os

ex = Experiment('ext_emb_gan')

@ex.config
def config():
    max_num_s = 2000000
    max_s_len = 20
    save_dir = os.path.join(cic.paths.DATA_DIR, 'ext_emb_gan')
    rnn_size = 500
    num_dsc_trains = 5
    num_epochs = 5
    num_generate = 20  # Number of sentences to generate after training
    gen_lr = .00001
    dsc_lr = .00001
    restore = False

@ex.automain
def main(max_num_s, max_s_len, save_dir, rnn_size, num_dsc_trains, num_epochs, num_generate, restore, gen_lr, dsc_lr):

    # Create save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load the Toronto dataset and instantiate random noise dataset
    ds = TorontoBookCorpus(max_s_len, result_path=cic.paths.BOOK_CORPUS_RESULT,
                           min_length=5, max_num_s=max_num_s, keep_unk_sentences=False,
                           vocab_min_freq=5, vocab=None, regenerate=False)
    inverted_vocab = {ds.vocab[k]: k for k in ds.vocab}

    sentences = convert_numpy_array_to_strings(ds.data[:num_generate, :], inverted_vocab, stop_token=ds.stop_token,
                                               keep_stop_token=False)
    for sentence in sentences:
        print(sentence)

    print('Num train sentences: %s' % len(ds))
    print('Len vocab: %s' % len(ds.vocab))

    rand_ds = GaussianRandomDataset(len(ds), rnn_size, 'z')

    # Load NLM embeddings
    embs_path = os.path.join(cic.paths.DATA_DIR, 'nlm_embs.npy')
    embs = np.load(open(embs_path, 'rb'))

    print('Mean embs: %s' % np.mean(embs))
    print('Var embs: %s' % np.var(embs))

    assert embs.shape[0] == len(ds.vocab)

    # Create Ext Emb GAN
    eegan = ExtEmbGAN(embs, max_s_len, rnn_size, num_dsc_trains, save_dir=save_dir, restore=restore)

    # Train
    if num_epochs > 0:
        eegan.train([ds, rand_ds], num_epochs=num_epochs, params={'gen_lr': gen_lr, 'dsc_lr': dsc_lr})

    # Generate sentences
    rand_pred_ds = GaussianRandomDataset(num_generate, rnn_size, 'z')
    pred_data = {'message': ds.data[:num_generate, :]}
    print('Pred data len: %s' % pred_data['message'].shape[0])
    result = eegan.predict([rand_pred_ds, pred_data], outputs=['gen_sentences', 'gen_embs'])

    np_gen_sentences = result['gen_sentences']
    gen_embs = result['gen_embs']
    print('Variance gen embs: %s' % np.var(gen_embs))

    gen_sentences = convert_numpy_array_to_strings(np_gen_sentences, inverted_vocab,
                                               stop_token=ds.stop_token, keep_stop_token=False)

    for index, sentence in enumerate(gen_sentences):

        print(np_gen_sentences[index, :])
        print(ds.data[index, :])
        print(sentence)
        print(sentences[index])
        print()

