"""Create a script to create an evaluation file for the NLM, VAE, and GAN models.
Volunteers edit this file to perform evaluation."""
import cic.config
import os
import pickle
import random

num_examples_per_model = 20

vae_sentences = pickle.load(open(os.path.join(cic.config.DATA_DIR, 'vae_messages.pkl'), 'rb'))
gan_sentences = pickle.load(open(os.path.join(cic.config.DATA_DIR, 'gan_messages.pkl'), 'rb'))
nlm_sentences = pickle.load(open(os.path.join(cic.config.DATA_DIR, 'nlm_messages.pkl'), 'rb'))

seed = 'hello world'

all_sentences = []
all_labels = []

num_vae_sentences = 0
for sentence in vae_sentences:
    sentence_tokens = sentence.split()

    if len(sentence_tokens) > 5 and len(sentence_tokens) <= 15:
        all_sentences.append(sentence)
        all_labels.append('vae')
        num_vae_sentences += 1

        if num_vae_sentences >= num_examples_per_model:
            break


num_gan_sentences = 0
for sentence in gan_sentences:
    sentence_tokens = sentence.split()

    if len(sentence_tokens) > 5 and len(sentence_tokens) <= 15:
        all_sentences.append(sentence)
        all_labels.append('gan')
        num_gan_sentences += 1

        if num_gan_sentences >= num_examples_per_model:
            break

num_nlm_sentences = 0
for sentence in nlm_sentences:
    sentence_tokens = sentence.split()

    if len(sentence_tokens) > 5 and len(sentence_tokens) <= 15:
        all_sentences.append(sentence)
        all_labels.append('nlm')
        num_nlm_sentences += 1

        if num_nlm_sentences >= num_examples_per_model:
            break

all_random_sentences = all_sentences.copy()
all_random_labels = all_labels.copy()
random.seed(seed)
random.shuffle(all_random_sentences)
random.seed(seed)
random.shuffle(all_random_labels)

for i in range(len(all_sentences)):
    each_sentence = all_sentences[i]
    for j in range(len(all_random_sentences)):
        each_random_sentence = all_random_sentences[j]
        if each_sentence == each_random_sentence:
            if not (all_labels[i] == all_random_labels[j]):
                print(each_sentence)
                print(each_random_sentence)
                print(all_labels[i])
                print(all_random_labels[j])
                raise AssertionError('Random shuffling is not right.')

assert len(all_sentences) == len(all_labels)
assert len(all_sentences) == 3 * num_examples_per_model

f = open(os.path.join(cic.config.DATA_DIR, 'evaluation'), 'wb')


print(all_sentences)
print(all_labels)
print(all_random_sentences)
print(all_random_labels)




