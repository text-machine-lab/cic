"""Model that uses the Cornell Movie Dialogs Corpus to conduct conversations. Here a conversation is a dialogue between
two characters represented by a sequence of messages."""
import numpy as np
import os
import spacy
import tensorflow as tf
from cic.question_answering import baseline_model_func, squad_dataset_tools as sdt

from cic import config
from cic.chat_bots import chat_model_func
from cic.chat_bots.chat_model_func import LEARNING_RATE, NUM_EXAMPLES_TO_PRINT, MAX_MESSAGE_LENGTH, \
    LEARNED_EMBEDDING_SIZE, \
    KEEP_PROB, \
    RNN_HIDDEN_DIM, TRAIN_FRACTION, BATCH_SIZE, NUM_EPOCHS, RESTORE_FROM_SAVE, REVERSE_INPUT_MESSAGE, STOP_TOKEN, \
    SEQ2SEQ_IMPLEMENTATION

# PRE-PROCESSING #######################################################################################################

if not os.path.exists(config.CHAT_MODEL_SAVE_DIR):
    os.makedirs(config.CHAT_MODEL_SAVE_DIR)

nlp = spacy.load('en')

examples, np_message, np_response, vocab_dict, vocabulary = chat_model_func.preprocess_all_cornell_conversations(nlp)

vocabulary_length = len(vocabulary)
num_examples = len(examples)

print(np_response.dtype)
print(np_response[:-10])

# GRAPH CREATION #######################################################################################################

print('Constructing model...')
tf_message = tf.placeholder(dtype=tf.int32, shape=[None, MAX_MESSAGE_LENGTH], name='input_message')
tf_response = tf.placeholder(dtype=tf.int32, shape=[None, MAX_MESSAGE_LENGTH], name='output_response')
tf_response_mask = tf.not_equal(tf_response, tf.constant(0), name='response_mask')
tf_keep_prob = tf.placeholder_with_default(1.0, (), name='keep_prob')
with tf.name_scope('batch_size'):
    tf_batch_size = tf.shape(tf_message)[0]
print('tf_response_mask shape: %s' % str(tf_response_mask.get_shape()))

tf_learned_embeddings = tf.get_variable('learned_embeddings',
                                        shape=[vocabulary_length, LEARNED_EMBEDDING_SIZE],
                                        initializer=tf.contrib.layers.xavier_initializer())

tf_message_embs = tf.nn.embedding_lookup(tf_learned_embeddings, tf_message, name='message_embeddings')

tf_message_embs = tf.nn.dropout(tf_message_embs, tf_keep_prob, name='message_embs_w_dropout')

print('Creating sequence-to-sequence...')

# Variables for transforming from message to response
tf_response_mapping_w1 = tf.get_variable('response_mapping_w1',
                                         shape=[RNN_HIDDEN_DIM, RNN_HIDDEN_DIM],
                                         initializer=tf.contrib.layers.xavier_initializer())
tf_response_mapping_b1 = tf.get_variable('response_mapping_b1',
                                         shape=[RNN_HIDDEN_DIM],
                                         initializer=tf.contrib.layers.xavier_initializer())
tf_response_mapping_w2 = tf.get_variable('response_mapping_w2',
                                         shape=[RNN_HIDDEN_DIM, RNN_HIDDEN_DIM],
                                         initializer=tf.contrib.layers.xavier_initializer())
tf_response_mapping_b2 = tf.get_variable('response_mapping_b2',
                                         shape=[RNN_HIDDEN_DIM],
                                         initializer=tf.contrib.layers.xavier_initializer())

if SEQ2SEQ_IMPLEMENTATION == 'dynamic_rnn':
    with tf.variable_scope('MESSAGE_ENCODER'):
        message_lstm = tf.contrib.rnn.LSTMCell(num_units=RNN_HIDDEN_DIM)
        #message_gru_reverse = tf.contrib.rnn.GRUCell(num_units=RNN_HIDDEN_DIM)
        tf_message_outputs, tf_message_state = tf.nn.dynamic_rnn(message_lstm, tf_message_embs, dtype=tf.float32)

    tf_latent_space_message = tf_message_outputs[:, -1, :]
    tf_latent_space_message_dropout = tf.nn.dropout(tf_latent_space_message, tf_keep_prob)
    tf_latent_space_middle = tf.nn.relu(tf.matmul(tf_latent_space_message_dropout, tf_response_mapping_w1) + tf_response_mapping_b1)
    tf_latent_space_response = tf.nn.relu(tf.matmul(tf_latent_space_middle, tf_response_mapping_w2) + tf_response_mapping_b2)

    tf_message_final_output_tile = tf.tile(tf.reshape(tf_latent_space_response, [-1, 1, RNN_HIDDEN_DIM]), [1,
                                                                                                           MAX_MESSAGE_LENGTH, 1])

    with tf.variable_scope('RESPONSE_DECODER'):
        response_lstm = tf.contrib.rnn.LSTMCell(num_units=RNN_HIDDEN_DIM)
        tf_response_outputs, tf_response_state = tf.nn.dynamic_rnn(response_lstm, tf_message_final_output_tile,
                                                                   dtype=tf.float32,
                                                                   initial_state=tf_message_state)
elif SEQ2SEQ_IMPLEMENTATION == 'homemade':
    with tf.variable_scope('MESSAGE_ENCODER'):
        message_lstm = tf.contrib.rnn.LSTMCell(num_units=RNN_HIDDEN_DIM)
        tf_message_state = message_lstm.zero_state(tf_batch_size, tf.float32)
        for lstm_step in range(MAX_MESSAGE_LENGTH):
            with tf.variable_scope('ENCODER_STEP') as message_scope:
                tf_message_input = tf.nn.embedding_lookup(tf_learned_embeddings, tf_message[:, lstm_step], name='message_timestep_input')
                tf_message_output, tf_message_state = message_lstm(tf_message_input, tf_message_state)

    tf_latent_space_message = tf_message_output
    tf_latent_space_message_dropout = tf.nn.dropout(tf_latent_space_message, tf_keep_prob)
    tf_latent_space_middle = tf.nn.relu(tf.matmul(tf_latent_space_message_dropout, tf_response_mapping_w1) + tf_response_mapping_b1)
    tf_latent_space_response = tf.nn.relu(tf.matmul(tf_latent_space_middle, tf_response_mapping_w2) + tf_response_mapping_b2)

    with tf.variable_scope('RESPONSE_DECODER'):
        response_lstm = tf.contrib.rnn.LSTMCell(num_units=RNN_HIDDEN_DIM)
        tf_response_state = tf_message_state  # response_lstm.zero_state(tf_batch_size, tf.float32)
        tf_response_output = tf.zeros([tf_batch_size, RNN_HIDDEN_DIM])
        response_outputs = []
        for lstm_step in range(MAX_MESSAGE_LENGTH):
            with tf.variable_scope('DECODER_STEP') as response_scope:
                tf_response_output, tf_response_state = response_lstm(tf_latent_space_response, tf_response_state)
                response_outputs.append(tf_response_output)
    tf_response_outputs = tf.stack(response_outputs, axis=1, name='response_outputs')

else:
    print('No sequence to sequence implementation specified. Exiting...')
    exit()


with tf.variable_scope('OUTPUT_PREDICTION'):
    print('Creating output layer...')
    W_v = tf.get_variable('output_weight',
                          shape=[RNN_HIDDEN_DIM, vocabulary_length],
                          initializer=tf.contrib.layers.xavier_initializer())
    W_b = tf.get_variable('output_bias',
                          shape=[vocabulary_length],
                          initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope('tf_response_log_probabilities'):
        tf_response_log_probabilities = tf.reshape(tf.matmul(tf.reshape(tf_response_outputs, [-1, RNN_HIDDEN_DIM]), W_v) + W_b,
                                                   [-1, MAX_MESSAGE_LENGTH, vocabulary_length])

    tf_response_probabilities = tf.nn.softmax(tf_response_log_probabilities, name='response_probabilities')

    tf_response_prediction = tf.argmax(tf_response_probabilities, axis=2)
    print('tf_response_prediction shape: %s' % str(tf_response_prediction.get_shape()))

with tf.variable_scope('LOSS'):
    # tf_response_log_probabilities_flat = tf.reshape(tf_response_log_probabilities, [-1, vocabulary_length])
    # tf_response_flat = tf.reshape(tf_response, [-1])
    tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_response_log_probabilities,
                                                               labels=tf_response,
                                                               name='word_losses')
    # tf_losses = tf.reshape(tf_losses_flat, [-1, MAX_MESSAGE_LENGTH])
    with tf.name_scope('total_loss'):
        tf_total_loss = tf.reduce_sum(tf_losses) / tf.cast(tf_batch_size, tf.float32)

baseline_model_func.create_tensorboard_visualization('chat')

with tf.name_scope("SAVER"):
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf_total_loss)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
# TRAINING #############################################################################################################

if RESTORE_FROM_SAVE:
    print('Restoring from save...')
    baseline_model_func.restore_model_from_save(config.CHAT_MODEL_SAVE_DIR, var_list=tf.trainable_variables(), sess=sess)

num_batches = int(num_examples * TRAIN_FRACTION / BATCH_SIZE)
num_train_examples = num_batches * BATCH_SIZE

if num_train_examples > 0 and NUM_EPOCHS > 0:
    np_train_message = np_message[:num_train_examples, :]
    np_train_response = np_response[:num_train_examples, :]
    train_examples = examples[:num_train_examples]
    for epoch in range(NUM_EPOCHS):
        print('Epoch: %s' % epoch)
        all_batch_losses = []
        all_batch_predictions = []
        for batch_index in range(num_batches):
            np_batch_message = np_train_message[batch_index * BATCH_SIZE:batch_index * BATCH_SIZE + BATCH_SIZE, :]
            np_batch_response = np_train_response[batch_index * BATCH_SIZE:batch_index * BATCH_SIZE + BATCH_SIZE, :]
            assert np_batch_message.shape == (BATCH_SIZE, MAX_MESSAGE_LENGTH)
            assert np_batch_response.shape == (BATCH_SIZE, MAX_MESSAGE_LENGTH)

            batch_loss, batch_response_predictions, _, batch_mask = sess.run([tf_total_loss, tf_response_prediction, train_op, tf_response_mask],
                                                                             feed_dict={tf_message: np_batch_message,
                                                                                        tf_response: np_batch_response,
                                                                                        tf_keep_prob: KEEP_PROB})
            all_batch_losses.append(batch_loss)
            all_batch_predictions.append(batch_response_predictions)
        print('Epoch loss: %s' % np.mean(all_batch_losses))
        saver.save(sess, config.CHAT_MODEL_SAVE_DIR, global_step=epoch)  # Save model after every epoch

    np_train_predictions = np.concatenate(all_batch_predictions, axis=0)

    train_predictions = sdt.convert_numpy_array_to_strings(np_train_predictions, vocabulary, stop_token=STOP_TOKEN)

    num_train_examples_correct = 0
    for i in range(len(train_predictions)):
        if train_predictions[i] + STOP_TOKEN == ' '.join(train_examples[i][1]):
            num_train_examples_correct += 1
    print('EM train accuracy: %s' % (num_train_examples_correct / num_train_examples))


    print('Printing training examples...')
    for i, each_prediction in enumerate(train_predictions):
        if i < NUM_EXAMPLES_TO_PRINT:
            print('Message: %s' % (' '.join(train_examples[i][0])))
            print('Label: %s' % (' '.join(train_examples[i][1])))
            print('Prediction: %s' % each_prediction)
            print('Message array: %s' % np_train_message[i, :].astype(int))
            print('Label array: %s' % np_train_response[i, :].astype(int))
            print('Prediction array: %s' % np_train_predictions[i, :].astype(int))

# PREDICTION ###########################################################################################################

if num_train_examples < num_examples:
    np_val_message = np_message[num_train_examples:, :]
    np_val_response = np_response[num_train_examples:, :]
    val_examples = examples[num_train_examples:]

    num_val_batches = int(np_val_message.shape[0] / BATCH_SIZE + 1)
    all_val_prediction_batches = []
    for batch_index in range(num_val_batches):
        np_val_message_batch = np_val_message[batch_index * BATCH_SIZE:batch_index * BATCH_SIZE + BATCH_SIZE, :]
        np_val_prediction_batch = sess.run(tf_response_prediction, feed_dict={tf_message: np_val_message_batch})
        all_val_prediction_batches.append(np_val_prediction_batch)

    np_val_predictions = np.concatenate(all_val_prediction_batches, axis=0)

    val_predictions = sdt.convert_numpy_array_to_strings(np_val_predictions, vocabulary, stop_token=STOP_TOKEN)

    num_val_examples_correct = 0
    for i in range(len(val_predictions)):
        if val_predictions[i] + STOP_TOKEN == ' '.join(val_examples[i][1]):
            num_val_examples_correct += 1
    print('EM validation accuracy: %s' % (num_val_examples_correct / (num_examples - num_train_examples)))

    print('\nPrinting validation examples...')
    for i, each_prediction in enumerate(val_predictions):
        if i < NUM_EXAMPLES_TO_PRINT:
            print('Message: %s' % (' '.join(val_examples[i][0])))
            print('Label: %s' % (' '.join(val_examples[i][1])))
            print('Prediction: %s' % each_prediction)
            print('Message array: %s' % np_val_message[i, :].astype(int))
            print('Label array: %s' % np_val_response[i, :].astype(int))
            print('Prediction array: %s' % np_val_predictions[i, :].astype(int))

# CHAT #################################################################################################################

print('\nChat with the chat bot! Enter a message:')
while True:
    chat_message = input('You: ')
    tk_chat_message = nlp.tokenizer(chat_message.lower())
    tk_chat_tokens = [str(token) for token in tk_chat_message if str(token) != ' ' and str(token) in vocab_dict] + [
        STOP_TOKEN]
    #print(tk_chat_tokens)
    np_chat_message = chat_model_func.construct_numpy_from_messages([tk_chat_tokens], vocab_dict, MAX_MESSAGE_LENGTH)
    if REVERSE_INPUT_MESSAGE:
        np_chat_message = np.flip(np_chat_message, axis=1)
    if num_train_examples < num_examples:
        for i in range(np_val_message.shape[0]):
            np_chat_message_flat = np.reshape(np_chat_message, [MAX_MESSAGE_LENGTH])
            if np.array_equal(np_chat_message_flat, np_val_message[i, :]):
                print(val_examples[i])
                print(np_val_message[i, :])
                print(np_chat_message_flat)
    print(np_chat_message)
    np_chat_response = sess.run(tf_response_prediction, feed_dict={tf_message: np_chat_message})
    print(np_chat_response)
    response = sdt.convert_numpy_array_to_strings(np_chat_response, vocabulary, stop_token=STOP_TOKEN)
    print('Bot: %s' % response[0])



