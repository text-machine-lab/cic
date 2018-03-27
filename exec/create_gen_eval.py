"""Create a script to create an evaluation file for the NLM, VAE, and GAN models.
Volunteers edit this file to perform evaluation."""
from cic.datasets.book_corpus import TorontoBookCorpus
from cic.datasets.text_dataset import convert_numpy_array_to_strings
import cic.paths
import os
import pickle
import random
import numpy as np

num_examples_per_model = 333
max_number_of_sentences = 2000000

vae_sentences = pickle.load(open(os.path.join(cic.paths.DATA_DIR, 'vae_messages.pkl'), 'rb'))
gan_sentences = pickle.load(open(os.path.join(cic.paths.DATA_DIR, 'gan_messages.pkl'), 'rb'))
nlm_sentences = pickle.load(open(os.path.join(cic.paths.DATA_DIR, 'nlm_messages.pkl'), 'rb'))

for index in range(len(vae_sentences)):
    vae_sentences[index] = ' '.join(vae_sentences[index].replace('<STOP>', '').split())

    if index < 10:
        print(vae_sentences[index])

ds = TorontoBookCorpus(20, result_path=cic.paths.BOOK_CORPUS_RESULT,
                       min_length=5, max_num_s=max_number_of_sentences, keep_unk_sentences=False,
                       vocab_min_freq=5, vocab=None, regenerate=False)

seed = 'hello world'

all_gen_sents = []
all_real_sents = []
all_labels = []

indices = np.arange(len(ds))
np.random.shuffle(indices)

# sent = sentence
# Grab sentences from each model: GAN, NLM, VAE
# Grab real sentences
# Form sentence pairs with real data and generated data from each model
# Randomly swap sentence pairs
# Create text file for all pairs (left, right)
# Create pickle file holding answers (model, left/right)

inverse_vocab = {ds.vocab[k]:k for k in ds.vocab}
for i in range(num_examples_per_model * 3):
    np_real_sent = np.reshape(ds[indices[i]]['message'], [1, -1])
    real_sent = convert_numpy_array_to_strings(np_real_sent, inverse_vocab, ds.stop_token, keep_stop_token=False)[0]
    all_real_sents.append(real_sent)

num_vae_sentences = 0
for sentence in vae_sentences:

    all_gen_sents.append(sentence)
    all_labels.append('vae')
    num_vae_sentences += 1

    if num_vae_sentences >= num_examples_per_model:
        break


num_gan_sentences = 0
for sentence in gan_sentences:

    all_gen_sents.append(sentence)
    all_labels.append('gan')
    num_gan_sentences += 1

    if num_gan_sentences >= num_examples_per_model:
        break

num_nlm_sentences = 0
for sentence in nlm_sentences:

    all_gen_sents.append(sentence)
    all_labels.append('nlm')
    num_nlm_sentences += 1

    if num_nlm_sentences >= num_examples_per_model:
        break

# Shuffle and sort
# rand = random

rand_gen_sents = all_gen_sents.copy()
rand_labels = all_labels.copy()
rand_real_sents = all_real_sents.copy()
random.seed(seed)
random.shuffle(rand_gen_sents)
random.seed(seed)
random.shuffle(rand_labels)
random.seed(seed)
random.shuffle(rand_real_sents)

print('Len rand_gen_sents: %s' % len(rand_gen_sents))
print('Len rand_labels: %s' % len(rand_labels))
print('Len rand_real_sents: %s' % len(rand_real_sents))

assert len(rand_gen_sents) == len(rand_labels)
assert len(rand_labels) == len(rand_real_sents)

f = open(os.path.join(cic.paths.DATA_DIR, 'evaluation.txt'), 'w')


record_gen_first = []  # this tracks, per line, if the generated example came first
output_pairs = []
for i in range(len(rand_gen_sents)):
    gen_first = random.choice([True, False])
    record_gen_first.append(gen_first)

    if gen_first:
        f.write("%s | %s\n" % (rand_gen_sents[i], rand_real_sents[i]))
        output_pairs.append([rand_gen_sents[i], rand_real_sents[i]])
    else:
        f.write("%s | %s\n" % (rand_real_sents[i], rand_gen_sents[i]))
        output_pairs.append([rand_real_sents[i], rand_gen_sents[i]])

answers = [(rand_label, gen_first) for rand_label, gen_first in zip(rand_labels, record_gen_first)]
pickle.dump(answers, open(os.path.join(cic.paths.DATA_DIR, 'evaluation_answers.pkl'), 'wb'))

# Check to make sure evaluation details work correctly...

assert len(output_pairs) == len(answers)
assert len(answers) == num_examples_per_model * 3

num_duplicates = 0
for index in range(len(output_pairs)):
    each_pair = output_pairs[index]
    left_s = each_pair[0]
    right_s = each_pair[1]
    each_answer = answers[index]
    label = each_answer[0]
    gen_first = each_answer[1]

    if gen_first:
        fake = left_s
        real = right_s
    else:
        fake = right_s
        real = left_s

    assert fake in rand_gen_sents
    assert fake in all_gen_sents
    assert real in all_real_sents
    assert real in rand_real_sents

    if fake in all_real_sents:
        num_duplicates += 1

    if label == 'vae':
        assert fake in vae_sentences
    elif label == 'gan':
        assert fake in gan_sentences
    elif label == 'nlm':
        assert fake in nlm_sentences
    else:
        raise ValueError('Wrong answer label')

print('Number of fake duplicates: %s' % num_duplicates)

# Print out some evaluation examples for further checking.
num_examples_print = 20

for index in range(len(output_pairs)):
    each_pair = output_pairs[index]
    left_s = each_pair[0]
    right_s = each_pair[1]
    each_answer = answers[index]
    label = each_answer[0]
    gen_first = each_answer[1]

    if index >= num_examples_print:
        break

    print('%s | %s | %s | %s' % (gen_first, left_s, right_s, label))

