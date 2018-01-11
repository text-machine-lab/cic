"""Holds conversations by processing messages and produces responses within the latent space created
by the auto-encoder trained in auto_encoder_model.py"""
import optparse
import pickle

import numpy as np
import os
import spacy
from cic.chat_bots import latent_chat
from cic.qa import squad_tools as sdt

from cic import config
from cic.ae import autoencoder as aef
from cic.chat_bots import chat_model


def parse_command_line_arguments():
    parser = optparse.OptionParser()
    parser.add_option('-t', '--train', dest="train", default=False, action='store_true', help='train model for the specified number of epochs, and save')
    parser.add_option('-s', '--save_dir', dest="save_dir", default=config.LATENT_CHAT_MODEL_SAVE_DIR, help='specify save directory for training and restoring')
    parser.add_option('-n', '--num_epochs', dest="num_epochs", default=latent_chat.NUM_EPOCHS, help='specify number of epochs to train for')
    parser.add_option('-a', '--ae_save_dir', dest="auto_encoder_save_dir", default=config.AUTO_ENCODER_MODEL_SAVE_DIR, help='specify save directory for auto-encoder model')
    parser.add_option('-r', '--restore_from_save', dest="restore_from_save", default=False, action='store_true', help='load model parameters from specified save directory')
    parser.add_option('-b', '--bot', dest="bot", default=False, action='store_true', help='talk with chat bot')
    parser.add_option('-v', '--variational', dest='variational', default=False, action='store_true', help='use variational loss')
    parser.add_option('-p', '--pre-process', dest='preprocess', default=False, action='store_true', help='true to perform pre-processing, false to load it from save')

    (options, args) = parser.parse_args()

    if not options.train:
        latent_chat.NUM_EPOCHS = 0
    else:
        latent_chat.NUM_EPOCHS = int(options.num_epochs)
    config.LATENT_CHAT_MODEL_SAVE_DIR = options.save_dir
    latent_chat.RESTORE_FROM_SAVE = options.restore_from_save

    aef.VARIATIONAL = options.variational

    latent_chat.PRE_PROCESS = options.preprocess

    config.AUTO_ENCODER_MODEL_SAVE_DIR = options.auto_encoder_save_dir
    config.AUTO_ENCODER_VOCAB_DICT = os.path.join(config.AUTO_ENCODER_MODEL_SAVE_DIR, 'vocab_dict.pkl')

    return options


def print_examples(examples):
    if latent_chat.NUM_EXAMPLES is None or latent_chat.NUM_EXAMPLES > 10:
        for i in range(10):
            print(examples[i])
    else:
        for example in examples:
            print(example)


def pre_process_latent_chat_model_data(vocab_dict, nlp):
    examples, np_message, np_response, _vocab_dict_, _vocabulary_ = chat_model.preprocess_all_cornell_conversations(
        nlp,
        vocab_dict=vocab_dict,
        reverse_inputs=False,
        max_message_length=aef.MAX_MSG_LEN)

    assert vocab_dict == _vocab_dict_
    assert vocabulary == _vocabulary_

    if latent_chat.NUM_EXAMPLES is not None:
        examples = examples[:latent_chat.NUM_EXAMPLES]
        np_message = np_message[:latent_chat.NUM_EXAMPLES, :]
        np_response = np_response[:latent_chat.NUM_EXAMPLES, :]

    print('Converting messages into latent space')
    np_latent_message = lcm.encoder.encode(np_message, aef.BATCH_SIZE)
    np_latent_response = lcm.encoder.encode(np_response, aef.BATCH_SIZE)
    print(np_latent_message.shape)
    print(np_latent_response.shape)

    pre_processing_dump_dict = {'examples': examples,
                                'message': np_message,
                                'response': np_response,
                                'vocab_dict': vocab_dict,
                                'vocabulary': vocabulary,
                                'latent_message': np_latent_message,
                                'latent_response': np_latent_response}

    return pre_processing_dump_dict


def validate_latent_chat_model_data():
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
    return num_messages_correctly_reconstructed, num_responses_correctly_reconstructed

# MAIN PROGRAM #########################################################################################################

options = parse_command_line_arguments()

print('Number of epochs: %s' % latent_chat.NUM_EPOCHS)
print('Model save directory: %s' % config.LATENT_CHAT_MODEL_SAVE_DIR)

print('Loading vocabulary...')
vocab_dict = pickle.load(open(config.AUTO_ENCODER_VOCAB_DICT, 'rb'))
vocabulary = sdt.invert_dictionary(vocab_dict)

lcm = latent_chat.LatentChatModel(len(vocab_dict), latent_chat.LEARNING_RATE,
                                  config.LATENT_CHAT_MODEL_SAVE_DIR,
                                  ae_save_dir=config.AUTO_ENCODER_MODEL_SAVE_DIR,
                                  restore_from_save=latent_chat.RESTORE_FROM_SAVE)

nlp = spacy.load('en')

if latent_chat.PRE_PROCESS:
    print('Pre-processing all input data...')
    pre_processing_dump_dict = pre_process_latent_chat_model_data(vocab_dict, nlp)
    print('Saving pre-processed data')
    pickle.dump(pre_processing_dump_dict, open(config.LATENT_CHAT_PRE_PROCESSING_DUMP, 'wb'))
else:
    print('Loading pre-processed data from save')
    pre_processing_dump_dict = pickle.load(open(config.LATENT_CHAT_PRE_PROCESSING_DUMP, 'rb'))

examples = pre_processing_dump_dict['examples']
np_message = pre_processing_dump_dict['message']
np_response = pre_processing_dump_dict['response']
np_latent_message = pre_processing_dump_dict['latent_message']
np_latent_response = pre_processing_dump_dict['latent_response']

print_examples(examples)

if latent_chat.VALIDATE_INPUTS:
    num_messages_correctly_reconstructed, num_responses_correctly_reconstructed = validate_latent_chat_model_data()

    print('Fraction of correctly encoded messages: %s' % (num_messages_correctly_reconstructed / len(examples)))
    print('Fraction of correctly encoded responses: %s' % (num_responses_correctly_reconstructed / len(examples)))

print(np.mean(np_latent_message))
print(np.std(np_latent_message))
print(np.max(np_latent_message))

np_mean_latent_response = np.mean(np_latent_response, axis=0)
mean_latent_loss = np.mean(np.square(np_latent_response - np_mean_latent_response))
print('Baseline mean latent loss (no access to input): %s' % mean_latent_loss)

lcm.train(np_latent_message, np_latent_response, latent_chat.NUM_EPOCHS, latent_chat.BATCH_SIZE, latent_chat.KEEP_PROB)

if latent_chat.CALCULATE_TRAIN_ACCURACY:
    num_samples_for_accuracy_prediction = 1000
    print('Predicting train accuracy...')
    np_train_latent_predictions = lcm.predict(np_latent_message[:num_samples_for_accuracy_prediction, :], latent_chat.BATCH_SIZE)
    np_train_predictions = lcm.decoder.decode(np_train_latent_predictions, latent_chat.BATCH_SIZE)
    train_predictions = sdt.convert_numpy_array_to_strings(np_train_predictions, vocabulary,
                                                           stop_token=aef.STOP_TOKEN,
                                                           keep_stop_token=True)
    num_correct_train_examples = 0
    for i in range(num_samples_for_accuracy_prediction):
        each_train_prediction = train_predictions[i]
        each_train_response = ' '.join(examples[i][1])
        print(each_train_prediction, '|', each_train_response)
        if each_train_prediction == each_train_response:
            num_correct_train_examples += 1
    frac_correct_train_examples = num_correct_train_examples / num_samples_for_accuracy_prediction
    print('EM Training Score: %s' % frac_correct_train_examples)

if options.bot:
    print('Speak to the model')
    while True:
        your_message = input('Message: ')
        if your_message == 'exit':
            break
        model_response = lcm.predict_string(your_message, nlp, vocab_dict, vocabulary)
        print('Response: %s' % model_response)

