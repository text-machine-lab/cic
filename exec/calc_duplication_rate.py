import pickle
import os
import cic.paths

num_examples_per_model = 100

vae_sentences = pickle.load(open(os.path.join(cic.paths.DATA_DIR, 'vae_messages.pkl'), 'rb'))[:num_examples_per_model]
gan_sentences = pickle.load(open(os.path.join(cic.paths.DATA_DIR, 'gan_messages.pkl'), 'rb'))[:num_examples_per_model]
nlm_sentences = pickle.load(open(os.path.join(cic.paths.DATA_DIR, 'nlm_messages.pkl'), 'rb'))[:num_examples_per_model]

def count_duplicates(sentences):
    with open(cic.paths.BOOK_CORPUS_P1) as f:
        for line in f:
            print(line)
