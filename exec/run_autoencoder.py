"""File to run vanilla autoencoder (no arcadian integration) on a vanilla Cornell Movie Dialogues dataset (no arcadian integration)."""
import optparse
import pickle

import gensim
import numpy as np
import os
import random
import spacy
import tensorflow as tf
from cic.utils import squad_tools as sdt, mdd_tools as mddt

from cic import config
from cic.models import old_autoencoder, old_chat_model, match_lstm
from cic.models.old_autoencoder import MAX_MSG_LEN, MAX_NUM_MSGS, STOP_TOKEN, RNN_HIDDEN_DIM, \
    LEARNED_EMB_DIM, LEARNING_RATE, KEEP_PROB, BATCH_SIZE, TRAINING_FRACTION, NUM_EPOCHS, \
    NUM_PRINT, VALIDATE, SAVE_TENSORBOARD, SHUFFLE

# ARGUMENTS ############################################################################################################
parser = optparse.OptionParser()
parser.add_option('-t', '--train', dest="train", default=False, action='store_true', help='train model for the specified number of epochs, and save')
parser.add_option('-s', '--save_dir', dest="save_dir", default=config.AUTO_ENCODER_MODEL_SAVE_DIR, help='specify save directory for training and restoring')
parser.add_option('-n', '--num_epochs', dest="num_epochs", default=NUM_EPOCHS, help='specify number of epochs to train for')
parser.add_option('-r', '--restore_from_save', dest="restore_from_save", default=False, action='store_true', help='load model parameters from specified save directory')
parser.add_option('-b', '--bot', dest="bot", default=False, action='store_true', help='test reconstruction of autoencoder')
#parser.add_option('-f', '--teacher_force', dest='teacher_force', default=False, action='store_true', help='use teacher forcing in decoder')
parser.add_option('-v', '--variational', dest='variational', default=False, action='store_true', help='use variational loss')
parser.add_option('-m', '--max_messages', dest='max_messages', default=MAX_NUM_MSGS, help='specify maximum number of messages to train and test on')

(options, args) = parser.parse_args()

if not options.train:
    NUM_EPOCHS = 0
else:
    NUM_EPOCHS = int(options.num_epochs)

config.AUTO_ENCODER_MODEL_SAVE_DIR = options.save_dir
config.AUTO_ENCODER_VOCAB_DICT = os.path.join(config.AUTO_ENCODER_MODEL_SAVE_DIR, 'vocab_dict.pkl')

if options.max_messages is not None:
    MAX_NUMBER_OF_MESSAGES = int(options.max_messages)

old_autoencoder.VARIATIONAL = options.variational
RESTORE_FROM_SAVE = options.restore_from_save

print('Number of epochs: %s' % NUM_EPOCHS)
print('Save directory: %s' % config.AUTO_ENCODER_MODEL_SAVE_DIR)

# PRE-PROCESSING #######################################################################################################

# # First cmd-line argument is save path for model checkpoint
# if len(sys.argv) >= 2:
#     config.AUTO_ENCODER_MODEL_SAVE_DIR = sys.argv[1]

if not os.path.exists(config.AUTO_ENCODER_MODEL_SAVE_DIR):
    os.makedirs(config.AUTO_ENCODER_MODEL_SAVE_DIR)

print('Loading nlp...')
nlp = spacy.load('en')

print('Loading messages...')
messages = mddt.load_messages_from_cornell_movie_lines(config.CORNELL_MOVIE_LINES_FILE, nlp,
                                                       max_number_of_messages=MAX_NUMBER_OF_MESSAGES,
                                                       max_message_length=MAX_MSG_LEN,
                                                       stop_token=STOP_TOKEN)
print('Number of Movie Dialogue messages: %s' % len(messages))
if old_autoencoder.USE_REDDIT:
    all_reddit_comments = []
    for filename in os.listdir(config.REDDIT_COMMENTS_DUMP):
        reddit_comment_file = open(os.path.join(config.REDDIT_COMMENTS_DUMP, filename), 'rb')
        reddit_comments = pickle.load(reddit_comment_file)
        all_reddit_comments += reddit_comments

    for i in range(10):
        print(all_reddit_comments[i])

    reddit_messages = [comment.split() for comment in all_reddit_comments if len(comment.split()) <= MAX_MSG_LEN]

    print('Number of Reddit messages: %s' % len(reddit_messages))

    # Combine Dialogue and Reddit comments
    messages += reddit_messages

if old_autoencoder.SEED is not None:
    random.seed(old_autoencoder.SEED)
if SHUFFLE:
    random.shuffle(messages)

np_message_lengths = np.array([len(message) for message in messages])
print('Average message length: %s' % np.mean(np_message_lengths))
print('Message length std: %s' % np.std(np_message_lengths))
print('Message max length: %s' % np.max(np_message_lengths))

if RESTORE_FROM_SAVE:
    print('Loading vocabulary from save')
    vocab_dict = pickle.load(open(config.AUTO_ENCODER_VOCAB_DICT, 'rb'))
    vocabulary = sdt.invert_dictionary(vocab_dict)
else:
    print('Building vocabulary')
    vocab_dict = gensim.corpora.Dictionary(documents=messages).token2id
    vocabulary = sdt.invert_dictionary(vocab_dict)
    # Add '' as index 0
    num_non_empty_words = len(vocab_dict)
    vocab_dict[vocabulary[0]] = num_non_empty_words
    vocab_dict[''] = 0
    vocabulary[num_non_empty_words] = vocabulary[0]
    vocabulary[0] = ''
    print('Saving vocabulary')
    pickle.dump(vocab_dict, open(config.AUTO_ENCODER_VOCAB_DICT, 'wb'))

vocabulary_length = len(vocab_dict)
print('Vocabulary size: %s' % vocabulary_length)
for i in range(10):
    print(i, vocabulary[i])

print('Constructing numpy array')
np_messages = old_chat_model.construct_numpy_from_messages(messages, vocab_dict, MAX_MSG_LEN)
num_messages = np_messages.shape[0]
print('np_messages shape: %s' % str(np_messages.shape))

print('Validating inputs...')
message_reconstruct = sdt.convert_numpy_array_to_strings(np_messages, vocabulary,
                                                         stop_token=STOP_TOKEN,
                                                         keep_stop_token=True)
for i in range(len(messages)):
    each_message = ' '.join(messages[i])

    if len(messages[i]) <= MAX_MSG_LEN:
        assert each_message == message_reconstruct[i]

# BUILD GRAPH ##########################################################################################################

print('Building model...')
with tf.Graph().as_default() as autoencoder_graph:
    auto_encoder = old_autoencoder.AutoEncoder(LEARNED_EMB_DIM, vocabulary_length, RNN_HIDDEN_DIM,
                                               MAX_MSG_LEN, encoder=True, decoder=True,
                                               save_dir=config.AUTO_ENCODER_MODEL_SAVE_DIR,
                                               load_from_save=RESTORE_FROM_SAVE,
                                               learning_rate=LEARNING_RATE,
                                               variational=old_autoencoder.VARIATIONAL,
                                               use_teacher_forcing=True)
    if SAVE_TENSORBOARD:
        match_lstm.create_tensorboard_visualization('chat')

# TRAIN ################################################################################################################

num_train_messages = int(num_messages * TRAINING_FRACTION)
if NUM_EPOCHS > 0:
    np_train_messages = np_messages[:num_train_messages, :]
    np_train_message_reconstruct = auto_encoder.train(np_train_messages, NUM_EPOCHS, BATCH_SIZE, keep_prob=KEEP_PROB)

    print('Printing train examples...')
    train_message_reconstruct = sdt.convert_numpy_array_to_strings(np_train_message_reconstruct, vocabulary,
                                                                   stop_token=STOP_TOKEN,
                                                                   keep_stop_token=True)
    assert np_train_messages.shape[0] == len(train_message_reconstruct)
    assert len(train_message_reconstruct) == num_train_messages

    num_train_examples_correct = 0
    for index, message_reconstruct in enumerate(train_message_reconstruct):
        original_message = ' '.join(messages[index])
        if message_reconstruct == original_message:
            num_train_examples_correct += 1
            #if index < NUM_EXAMPLES_TO_PRINT:
        print(message_reconstruct, '\t\t\t|||', original_message)

    print('Training EM Accuracy: %s' % (num_train_examples_correct / num_train_messages))

# PREDICT ##############################################################################################################

np_val_messages = np_messages[num_train_messages:, :]
num_val_messages = np_val_messages.shape[0]

np_val_message_reconstruct = auto_encoder.reconstruct(np_val_messages, BATCH_SIZE)

print('Printing validation examples...')
val_message_reconstruct = sdt.convert_numpy_array_to_strings(np_val_message_reconstruct, vocabulary,
                                                             stop_token=STOP_TOKEN,
                                                             keep_stop_token=True)
print(num_val_messages)
print(len(val_message_reconstruct))
print(np_val_message_reconstruct.shape)
print(np_val_messages.shape)
num_validation_examples_correct = 0
for index, message_reconstruct in enumerate(val_message_reconstruct):
    original_message = ' '.join(messages[num_train_messages + index])
    if index < NUM_PRINT:
        print(message_reconstruct, '\t\t\t|||', original_message)
    if original_message == message_reconstruct:
        num_validation_examples_correct += 1

validation_accuracy = num_validation_examples_correct / num_val_messages
print('Validation EM accuracy: %s' % validation_accuracy)

with tf.Graph().as_default() as encoder_graph:
    encoder = auto_encoder.AutoEncoder(LEARNED_EMB_DIM, vocabulary_length, RNN_HIDDEN_DIM,
                                       MAX_MSG_LEN, encoder=True, decoder=False,
                                       save_dir=config.AUTO_ENCODER_MODEL_SAVE_DIR,
                                       load_from_save=RESTORE_FROM_SAVE or NUM_EPOCHS > 0,
                                       learning_rate=LEARNING_RATE,
                                       variational=auto_encoder.VARIATIONAL)

with tf.Graph().as_default() as decoder_graph:
    decoder = auto_encoder.AutoEncoder(LEARNED_EMB_DIM, vocabulary_length, RNN_HIDDEN_DIM,
                                       MAX_MSG_LEN, encoder=False, decoder=True,
                                       save_dir=config.AUTO_ENCODER_MODEL_SAVE_DIR,
                                       load_from_save=RESTORE_FROM_SAVE or NUM_EPOCHS > 0,
                                       learning_rate=LEARNING_RATE,
                                       variational=auto_encoder.VARIATIONAL)

if VALIDATE:
    np_val_latent = encoder.encode(np_val_messages, BATCH_SIZE)
    val_latent_avg_magnitude = np.mean(np.abs(np_val_latent))
    val_latent_std = np.std(np_val_latent)
    np_val_decoder_reconstruct = decoder.decode(np_val_latent, BATCH_SIZE)
    print('Average magnitude of latent space dimension: %s' % val_latent_avg_magnitude)
    assert np.isclose(np_val_message_reconstruct, np_val_decoder_reconstruct).all()

# INTERACT #############################################################################################################

if options.bot:
    print('Test the autoencoder!')
    while(True):
        print('Would you like to test individual messages, or test the space? (individual/space/neither)')
        choice = input()
        if choice == 'individual':
            while True:
                your_message = input('Message: ')
                if your_message == 'exit':
                    break
                np_your_message = auto_encoder.convert_string_to_numpy(your_message, nlp, vocab_dict)
                np_your_message_reconstruct = auto_encoder.reconstruct(np_your_message, 1)
                your_message_reconstruct = sdt.convert_numpy_array_to_strings(np_your_message_reconstruct, vocabulary,
                                                                              stop_token=STOP_TOKEN,
                                                                              keep_stop_token=True)
                print('Reconstruction: %s' % your_message_reconstruct[0])

        if choice == 'space':
            while True:
                wish_to_continue = input('Continue? (y/n)')
                if wish_to_continue != 'y' and wish_to_continue != 'yes':
                    break
                num_increments = int(input('Number of INCREMENTS: '))
                first_message = input('First message: ')
                second_message = input('Second message: ')
                np_first_message = auto_encoder.convert_string_to_numpy(first_message, nlp, vocab_dict)
                np_second_message = auto_encoder.convert_string_to_numpy(second_message, nlp, vocab_dict)
                np_first_latent = encoder.encode(np_first_message, BATCH_SIZE)
                np_second_latent = encoder.encode(np_second_message, BATCH_SIZE)
                np_increment = (np_first_latent - np_second_latent) / num_increments
                for i in range(num_increments + 1):
                    np_new_latent = np_second_latent + i * np_increment
                    np_new_message = decoder.decode(np_new_latent, BATCH_SIZE)
                    new_message = sdt.convert_numpy_array_to_strings(np_new_message, vocabulary,
                                                                     stop_token=STOP_TOKEN,
                                                                     keep_stop_token=True)[0]
                    print('Increment %s: %s' % (i, new_message))

        if choice == 'neither':
            break



