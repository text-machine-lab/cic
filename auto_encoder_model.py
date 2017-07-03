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
import tensorflow as tf
import pickle

MAX_MESSAGE_LENGTH = 10
MAX_NUMBER_OF_MESSAGES = None
STOP_TOKEN = '<STOP>'
DELIMITER = ' +++$+++ '
RNN_HIDDEN_DIM = 1000
LEARNED_EMBEDDING_SIZE = 100
LEARNING_RATE = .0004
RESTORE_FROM_SAVE = True
BATCH_SIZE = 20
TRAINING_FRACTION = 0.8
NUM_EPOCHS = 0
NUM_EXAMPLES_TO_PRINT = 20

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

print('Building model...')
auto_encoder = auto_encoder_func.AutoEncoder(LEARNED_EMBEDDING_SIZE, vocabulary_length, RNN_HIDDEN_DIM,
                                             MAX_MESSAGE_LENGTH, encoder=True, decoder=True)

tf_message = tf.placeholder(dtype=tf.int32, shape=[None, MAX_MESSAGE_LENGTH], name='input_message')
with tf.name_scope('batch_size'):
    tf_batch_size = tf.shape(tf_message)[0]

tf_message_output = auto_encoder.build_encoder(tf_message)
tf_message_prediction, tf_message_log_prob, tf_message_prob = auto_encoder.build_decoder(tf_message_output)
tf_total_loss = auto_encoder.build_trainer(tf_message_log_prob, tf_message)
baseline_model_func.create_tensorboard_visualization('chat')

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf_total_loss)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

if RESTORE_FROM_SAVE:
    print('Restoring from save...')
    auto_encoder.load_encoder_from_save(config.AUTO_ENCODER_MODEL_SAVE_DIR, sess)
    auto_encoder.load_decoder_from_save(config.AUTO_ENCODER_MODEL_SAVE_DIR, sess)

with tf.name_scope("SAVER"):
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)

num_train_messages = int(num_messages * TRAINING_FRACTION)

if NUM_EPOCHS > 0:
    np_train_messages = np_messages[:num_train_messages, :]

    for epoch in range(NUM_EPOCHS):
        train_batch_gen = chat_model_func.BatchGenerator(np_train_messages, BATCH_SIZE)
        all_train_message_batches = []
        for batch_index, np_message_batch in enumerate(train_batch_gen.generate_batches()):
            _, batch_loss, np_batch_message_reconstruct = sess.run([train_op, tf_total_loss, tf_message_prediction],
                                                                   feed_dict={tf_message: np_message_batch})
            all_train_message_batches.append(np_batch_message_reconstruct)
            if batch_index % 200 == 0:
                print('Batch loss: %s' % batch_loss)
        saver.save(sess, config.AUTO_ENCODER_MODEL_SAVE_DIR, global_step=epoch)
    np_train_message_reconstruct = np.concatenate(all_train_message_batches, axis=0)

    print('Printing train examples...')
    train_message_reconstruct = sdt.convert_numpy_array_to_strings(np_train_message_reconstruct, vocabulary,
                                                                   stop_token=STOP_TOKEN,
                                                                   keep_stop_token=True)
    for index, message_reconstruct in enumerate(train_message_reconstruct[:NUM_EXAMPLES_TO_PRINT]):
        print(message_reconstruct, '\t\t\t|||', ' '.join(messages[index]))

np_val_messages = np_messages[num_train_messages:, :]
num_val_messages = np_val_messages.shape[0]

all_val_message_batches = []
val_batch_gen = chat_model_func.BatchGenerator(np_val_messages, BATCH_SIZE)
for batch_index, np_message_batch in enumerate(val_batch_gen.generate_batches()):
    np_val_batch_reconstruct = sess.run(tf_message_prediction, feed_dict={tf_message: np_message_batch})
    all_val_message_batches.append(np_val_batch_reconstruct)
np_val_message_reconstruct = np.concatenate(all_val_message_batches, axis=0)

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

print('Test the autoencoder!')
print('Would you like to test individual messages, or test the space? (individual/space/neither)')
choice = input()
if choice == 'individual':
    while True:
        your_message = input('Message: ')
        tk_message = nlp.tokenizer(your_message.lower())
        tk_tokens = [str(token) for token in tk_message if str(token) != ' ' and str(token) in vocab_dict] + [STOP_TOKEN]
        np_your_message = chat_model_func.construct_numpy_from_messages([tk_tokens], vocab_dict, MAX_MESSAGE_LENGTH)
        np_your_message_reconstruct = sess.run(tf_message_prediction, feed_dict={tf_message: np_your_message})
        your_message_reconstruct = sdt.convert_numpy_array_to_strings(np_your_message_reconstruct, vocabulary,
                                                                      stop_token=STOP_TOKEN,
                                                                      keep_stop_token=True)
        print('Reconstruction: %s' % your_message_reconstruct[0])
