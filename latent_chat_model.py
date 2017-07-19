"""Holds conversations by processing messages and produces responses within the latent space created
by the auto-encoder trained in auto_encoder_model.py"""
import optparse
import os
import pickle

import numpy as np
import spacy

import auto_encoder_func as aef
import chat_model_func
import config
import latent_chat_func
import squad_dataset_tools as sdt
from latent_chat_func import NUM_EXAMPLES, NUM_EPOCHS, VALIDATE_INPUTS

parser = optparse.OptionParser()
parser.add_option('-t', '--train', dest="train", default=False, help='train model for the specified number of epochs, and save')
parser.add_option('-s', '--save_dir', dest="save_dir", default=config.LATENT_CHAT_MODEL_SAVE_DIR, help='specify save directory for training and restoring')
parser.add_option('-n', '--num_epochs', dest="num_epochs", default=NUM_EPOCHS, help='specify number of epochs to train for')
parser.add_option('-a', '--ae_save_dir', dest="auto_encoder_save_dir", default=config.AUTO_ENCODER_MODEL_SAVE_DIR, help='specify save directory for auto-encoder model')
parser.add_option('-r', '--restore_from_save', dest="restore_from_save", default=False, help='load model parameters from specified save directory')
parser.add_option('-b', '--bot', dest="bot", default=False, help='talk with chat bot')

(options, args) = parser.parse_args()

if not options.train:
    NUM_EPOCHS = 0
else:
    NUM_EPOCHS = int(options.num_epochs)
config.LATENT_CHAT_MODEL_SAVE_DIR = options.save_dir
RESTORE_FROM_SAVE = options.restore_from_save

config.AUTO_ENCODER_MODEL_SAVE_DIR = options.auto_encoder_save_dir
config.AUTO_ENCODER_VOCAB_DICT = os.path.join(config.AUTO_ENCODER_MODEL_SAVE_DIR, 'vocab_dict.pkl')

print('Number of epochs: %s' % NUM_EPOCHS)
print('Model save directory: %s' % config.LATENT_CHAT_MODEL_SAVE_DIR)

print('Loading vocabulary...')
vocab_dict = pickle.load(open(config.AUTO_ENCODER_VOCAB_DICT, 'rb'))

nlp = spacy.load('en')

examples, np_message, np_response, vocab_dict, vocabulary = chat_model_func.preprocess_all_cornell_conversations(nlp,
                                                                                                                 vocab_dict=vocab_dict,
                                                                                                                 reverse_inputs=False)

if NUM_EXAMPLES is not None:
    examples = examples[:NUM_EXAMPLES]
    np_message = np_message[:NUM_EXAMPLES, :]
    np_response = np_response[:NUM_EXAMPLES, :]

if NUM_EXAMPLES is None or NUM_EXAMPLES > 10:
    for i in range(10):
        print(examples[i])
else:
    for example in examples:
        print(example)

num_examples = len(examples)


lcm = latent_chat_func.LatentChatModel(len(vocab_dict), latent_chat_func.LEARNING_RATE,
                                       config.LATENT_CHAT_MODEL_SAVE_DIR,
                                       restore_from_save=RESTORE_FROM_SAVE)

print('Converting messages into latent space')
np_latent_message = lcm.encoder.encode(np_message, aef.BATCH_SIZE)
np_latent_response = lcm.encoder.encode(np_response, aef.BATCH_SIZE)
print(np_latent_message.shape)
print(np_latent_response.shape)

if VALIDATE_INPUTS:
    print('Reconstructing messages from latent space')
    np_message_reconstruct = lcm.decoder.decode(np_latent_message, aef.BATCH_SIZE)
    np_response_reconstruct = lcm.decoder.decode(np_latent_response, aef.BATCH_SIZE)
    print(np_message_reconstruct.shape)
    print(np_response_reconstruct.shape)

    message_reconstruct = sdt.convert_numpy_array_to_strings(np_message_reconstruct, vocabulary,
                                                                      stop_token=aef.STOP_TOKEN,
                                                                      keep_stop_token=True)
    response_reconstruct = sdt.convert_numpy_array_to_strings(np_response_reconstruct, vocabulary,
                                                                      stop_token=aef.STOP_TOKEN,
                                                                      keep_stop_token=True)

    print('Shape np_latent_message: %s' % str(np_latent_message.shape))
    print('Shape np_latent_response: %s' % str(np_latent_response.shape))
    print(len(message_reconstruct))
    print(len(response_reconstruct))
    assert len(message_reconstruct) == len(response_reconstruct)
    num_messages_correctly_reconstructed = 0
    num_responses_correctly_reconstructed = 0
    for i in range(len(message_reconstruct)):
        original_message = ' '.join(examples[i][0])
        original_response = ' '.join(examples[i][1])
        if message_reconstruct[i] == original_message:
            num_messages_correctly_reconstructed += 1
        if response_reconstruct[i] == original_response:
            num_responses_correctly_reconstructed += 1
        if i < 5:
            print('Message: ' + message_reconstruct[i] + '|||' + original_message)
            print('Response: ' + response_reconstruct[i] + '|||' + original_response)

    print('Fraction of correctly encoded messages: %s' % (num_messages_correctly_reconstructed / num_examples))
    print('Fraction of correctly encoded responses: %s' % (num_responses_correctly_reconstructed / num_examples))

print(np.mean(np_latent_message))
print(np.std(np_latent_message))
print(np.max(np_latent_message))

np_mean_latent_response = np.mean(np_latent_response, axis=0)
mean_latent_loss = np.mean(np.square(np_latent_response - np_mean_latent_response))
print('Baseline mean latent loss (no access to input): %s' % mean_latent_loss)

lcm.train(np_latent_message, np_latent_response, NUM_EPOCHS, latent_chat_func.BATCH_SIZE, latent_chat_func.KEEP_PROB)

if options.bot:
    print('Speak to the model')
    while True:
        your_message = input('Message: ')
        if your_message == 'exit':
            break
        model_response = lcm.predict_string(your_message, nlp, vocab_dict, vocabulary)
        print('Response: %s' % model_response)

