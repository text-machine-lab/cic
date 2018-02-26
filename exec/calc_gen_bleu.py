"""Calculates bleu score for GAN, VAE, and NLM models."""
import pickle
import cic.config
import os
from cic.datasets.book_corpus import TorontoBookCorpus
import numpy as np
from cic.datasets.text_dataset import convert_numpy_array_to_strings
from nltk.translate.bleu_score import sentence_bleu
import tqdm

num_examples_per_model = 1000
max_number_of_sentences = 10000
regenerate=True
result_path = os.path.join(cic.config.DATA_DIR, 'validation_book_corpus/')

vae_sentences = pickle.load(open(os.path.join(cic.config.DATA_DIR, 'vae_messages.pkl'), 'rb'))[:num_examples_per_model]
gan_sentences = pickle.load(open(os.path.join(cic.config.DATA_DIR, 'gan_messages.pkl'), 'rb'))[:num_examples_per_model]
nlm_sentences = pickle.load(open(os.path.join(cic.config.DATA_DIR, 'nlm_messages.pkl'), 'rb'))[:num_examples_per_model]

# Remove stop token from VAE sentences. It gives them away!
for index in range(len(vae_sentences)):
    vae_sentences[index] = ' '.join(vae_sentences[index].replace('<STOP>', '').split())

    if index < 10:
        print(vae_sentences[index])

# Create validation set
ds = TorontoBookCorpus(20, result_path=result_path,
                       min_length=5, max_num_s=2000000, keep_unk_sentences=False,
                       vocab_min_freq=5, vocab=None, regenerate=regenerate, second_file_first=True)

indices = np.arange(len(ds))
np.random.shuffle(indices)

print('Converting from numpy to strings')
real_sentences = []
inverse_vocab = {ds.vocab[k]:k for k in ds.vocab}
for i in range(max_number_of_sentences):
    np_real_sent = np.reshape(ds[indices[i]]['message'], [1, -1])
    real_sent = convert_numpy_array_to_strings(np_real_sent, inverse_vocab, ds.stop_token, keep_stop_token=False)[0]
    # We split sentence into tokens before adding - required for bleu
    real_sentences.append(real_sent.split())

# now we have vae sentences, gan sentences, nlm sentences, and real_sentences
print(len(vae_sentences))
print(len(gan_sentences))
print(len(nlm_sentences))
print(len(real_sentences))

assert len(vae_sentences) == len(gan_sentences)
assert len(gan_sentences) == len(nlm_sentences)

# split each sentence on token
vae_s_lens = []
gan_s_lens = []
nlm_s_lens = []

for index in range(len(vae_sentences)):
    vae_sentences[index] = vae_sentences[index].split()
    vae_s_lens.append(len(vae_sentences[index]))

    gan_sentences[index] = gan_sentences[index].split()
    gan_s_lens.append(len(gan_sentences[index]))

    nlm_sentences[index] = nlm_sentences[index].split()
    nlm_s_lens.append(len(nlm_sentences[index]))

print('Avg len vae sentences: %s' % np.mean(vae_s_lens))
print('Avg len gan sentences: %s' % np.mean(gan_s_lens))
print('Avg len nlm sentences: %s' % np.mean(nlm_s_lens))

print('Calculating bleu scores')
vae_sentence_bleus = []
for sentence in tqdm.tqdm(vae_sentences):
    s_bleu = sentence_bleu(real_sentences, sentence)
    vae_sentence_bleus.append(s_bleu)
vae_bleu = np.mean(vae_sentence_bleus)

print('VAE Bleu: %s' % vae_bleu)

gan_sentence_bleus = []
for sentence in tqdm.tqdm(gan_sentences):
    s_bleu = sentence_bleu(real_sentences, sentence)
    gan_sentence_bleus.append(s_bleu)
gan_bleu = np.mean(gan_sentence_bleus)

print('GAN Bleu: %s' % gan_bleu)

nlm_sentence_bleus = []
for sentence in tqdm.tqdm(nlm_sentences):
    s_bleu = sentence_bleu(real_sentences, sentence)
    nlm_sentence_bleus.append(s_bleu)
nlm_bleu = np.mean(nlm_sentence_bleus)

print('NLM Bleu: %s' % nlm_bleu)

