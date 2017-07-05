"""Converts a sentence of words into a latent representation, then converts that representation
back into the original sentence, with some reconstruction loss. Latent space is continuous, and can
be used in a GAN setting."""
import chat_model_func
import squad_dataset_tools as sdt
import movie_dialogue_dataset_tools as mddt
import baseline_model_func
import auto_encoder_func
import spacy
import config
import numpy as np
import gensim
import pickle
import tensorflow as tf

# CONTROL PANEL ########################################################################################################

MAX_MESSAGE_LENGTH = 10
MAX_NUMBER_OF_MESSAGES = None
STOP_TOKEN = '<STOP>'
DELIMITER = ' +++$+++ '
RNN_HIDDEN_DIM = 1500
LEARNED_EMBEDDING_SIZE = 100
LEARNING_RATE = .0008
RESTORE_FROM_SAVE = True
BATCH_SIZE = 20
TRAINING_FRACTION = 0.8
NUM_EPOCHS = 50
NUM_EXAMPLES_TO_PRINT = 20
VALIDATE_ENCODER_AND_DECODER = False
SAVE_TENSORBOARD_VISUALIZATION = False

# PRE-PROCESSING #######################################################################################################

print('Loading nlp...')
nlp = spacy.load('en')

print('Loading messages...')
messages = mddt.load_messages_from_cornell_movie_lines(config.CORNELL_MOVIE_LINES_FILE, nlp,
                                                       max_number_of_messages=MAX_NUMBER_OF_MESSAGES,
                                                       max_message_length=MAX_MESSAGE_LENGTH,
                                                       stop_token=STOP_TOKEN)

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
np_messages = chat_model_func.construct_numpy_from_messages(messages, vocab_dict, MAX_MESSAGE_LENGTH)
num_messages = np_messages.shape[0]
print('np_messages shape: %s' % str(np_messages.shape))

print('Validating inputs...')
message_reconstruct = sdt.convert_numpy_array_to_strings(np_messages, vocabulary,
                                                         stop_token=STOP_TOKEN,
                                                         keep_stop_token=True)
for i in range(len(messages)):
    each_message = ' '.join(messages[i])

    if len(messages[i]) <= MAX_MESSAGE_LENGTH:
        assert each_message == message_reconstruct[i]

# BUILD GRAPH ##########################################################################################################

print('Building model...')
with tf.Graph().as_default() as autoencoder_graph:
    auto_encoder = auto_encoder_func.AutoEncoder(LEARNED_EMBEDDING_SIZE, vocabulary_length, RNN_HIDDEN_DIM,
                                                 MAX_MESSAGE_LENGTH, encoder=True, decoder=True,
                                                 save_dir=config.AUTO_ENCODER_MODEL_SAVE_DIR,
                                                 load_from_save=RESTORE_FROM_SAVE,
                                                 learning_rate=LEARNING_RATE)
    if SAVE_TENSORBOARD_VISUALIZATION:
        baseline_model_func.create_tensorboard_visualization('chat')

# TRAIN ################################################################################################################

num_train_messages = int(num_messages * TRAINING_FRACTION)
if NUM_EPOCHS > 0:
    np_train_messages = np_messages[:num_train_messages, :]
    np_train_message_reconstruct = auto_encoder.train(np_train_messages, NUM_EPOCHS, BATCH_SIZE)

    print('Printing train examples...')
    train_message_reconstruct = sdt.convert_numpy_array_to_strings(np_train_message_reconstruct, vocabulary,
                                                                   stop_token=STOP_TOKEN,
                                                                   keep_stop_token=True)
    for index, message_reconstruct in enumerate(train_message_reconstruct[:NUM_EXAMPLES_TO_PRINT]):
        print(message_reconstruct, '\t\t\t|||', ' '.join(messages[index]))

# PREDICT ##############################################################################################################

np_val_messages = np_messages[num_train_messages:, :]
num_val_messages = np_val_messages.shape[0]

np_val_message_reconstruct = auto_encoder.reconstruct(np_val_messages, BATCH_SIZE)

print('Printing validation examples...')
val_message_reconstruct = sdt.convert_numpy_array_to_strings(np_val_message_reconstruct, vocabulary,
                                                             stop_token=STOP_TOKEN,
                                                             keep_stop_token=True)

num_validation_examples_correct = 0
for index, message_reconstruct in enumerate(val_message_reconstruct[:NUM_EXAMPLES_TO_PRINT]):
    original_message = ' '.join(messages[num_train_messages + index])
    print(message_reconstruct, '\t\t\t|||', original_message)
    if original_message == message_reconstruct:
        num_validation_examples_correct += 1

validation_accuracy = num_validation_examples_correct / num_val_messages
print('Validation EM accuracy: %s' % validation_accuracy)

with tf.Graph().as_default() as encoder_graph:
    encoder = auto_encoder_func.AutoEncoder(LEARNED_EMBEDDING_SIZE, vocabulary_length, RNN_HIDDEN_DIM,
                                            MAX_MESSAGE_LENGTH, encoder=True, decoder=False,
                                            save_dir=config.AUTO_ENCODER_MODEL_SAVE_DIR,
                                            load_from_save=True,
                                            learning_rate=LEARNING_RATE)

with tf.Graph().as_default() as decoder_graph:
    decoder = auto_encoder_func.AutoEncoder(LEARNED_EMBEDDING_SIZE, vocabulary_length, RNN_HIDDEN_DIM,
                                            MAX_MESSAGE_LENGTH, encoder=False, decoder=True,
                                            save_dir=config.AUTO_ENCODER_MODEL_SAVE_DIR,
                                            load_from_save=True,
                                            learning_rate=LEARNING_RATE)

if VALIDATE_ENCODER_AND_DECODER:
    np_val_latent = encoder.encode(np_val_messages, BATCH_SIZE)
    np_val_decoder_reconstruct = decoder.decode(np_val_latent, BATCH_SIZE)

    assert np.isclose(np_val_message_reconstruct, np_val_decoder_reconstruct).all()

# INTERACT #############################################################################################################


def convert_string_to_numpy(msg):
    tk_message = nlp.tokenizer(msg.lower())
    tk_tokens = [str(token) for token in tk_message if str(token) != ' ' and str(token) in vocab_dict] + [STOP_TOKEN]
    np_message = chat_model_func.construct_numpy_from_messages([tk_tokens], vocab_dict, MAX_MESSAGE_LENGTH)
    return np_message

print('Test the autoencoder!')
print('Would you like to test individual messages, or test the space? (individual/space/neither)')
choice = input()
if choice == 'individual':
    while True:
        your_message = input('Message: ')
        np_your_message = convert_string_to_numpy(your_message)
        np_your_message_reconstruct = auto_encoder.reconstruct(np_your_message, 1)
        your_message_reconstruct = sdt.convert_numpy_array_to_strings(np_your_message_reconstruct, vocabulary,
                                                                      stop_token=STOP_TOKEN,
                                                                      keep_stop_token=True)
        print('Reconstruction: %s' % your_message_reconstruct[0])

if choice == 'space':
    while True:
        num_increments = int(input('Number of increments: '))
        first_message = input('First message: ')
        second_message = input('Second message: ')
        np_first_message = convert_string_to_numpy(first_message)
        np_second_message = convert_string_to_numpy(second_message)
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



