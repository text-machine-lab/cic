"""Converts a sentence of words into a latent representation, then converts that representation
back into the original sentence, with some reconstruction loss. Latent space is continuous, and can
be used in a GAN setting."""
import chat_model_func
import squad_dataset_tools as sdt
import baseline_model_func
import spacy
import config
import numpy as np
import gensim
import tensorflow as tf

MAX_MESSAGE_LENGTH = 10
MAX_NUMBER_OF_MESSAGES = None
STOP_TOKEN = '<STOP>'
DELIMITER = ' +++$+++ '
RNN_HIDDEN_DIM = 1000
LEARNED_EMBEDDING_SIZE = 100
LEARNING_RATE = .0005
RESTORE_FROM_SAVE = False
BATCH_SIZE = 20
TRAINING_FRACTION = 0.8
NUM_EPOCHS = 100
NUM_EXAMPLES_TO_PRINT = 20

print('Loading nlp...')
nlp = spacy.load('en')

print('Loading messages...')
movie_lines_file = open(config.CORNELL_MOVIE_LINES_FILE, 'rb')
messages = []
line_index = 0
for message_line in movie_lines_file:
    if MAX_NUMBER_OF_MESSAGES is None or line_index < MAX_NUMBER_OF_MESSAGES:
        try:
            message_data = message_line.decode('utf-8').split(DELIMITER)
            message_id = message_data[0]
            character_id = message_data[1]
            movie_id = message_data[2]
            character_name = message_data[3]
            message = message_data[4][:-1]
            tk_message = nlp.tokenizer(message.lower())
            tk_tokens = [str(token) for token in tk_message if str(token) != ' '] + [STOP_TOKEN]
            if len(tk_tokens) <= MAX_MESSAGE_LENGTH:
                messages.append(tk_tokens)
                line_index += 1
        except UnicodeDecodeError:
            pass
    else:
        break

np_message_lengths = np.array([len(message) for message in messages])
print('Average message length: %s' % np.mean(np_message_lengths))
print('Message length std: %s' % np.std(np_message_lengths))
print('Message max length: %s' % np.max(np_message_lengths))

print('Building vocabulary')
vocab_dict = gensim.corpora.Dictionary(documents=messages).token2id
vocabulary = sdt.invert_dictionary(vocab_dict)
vocabulary_length = len(vocab_dict)
print('Vocabulary size: %s' % vocabulary_length)

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
tf_message = tf.placeholder(dtype=tf.int32, shape=[None, MAX_MESSAGE_LENGTH], name='input_message')
with tf.name_scope('batch_size'):
    tf_batch_size = tf.shape(tf_message)[0]
tf_learned_embeddings = tf.get_variable('learned_embeddings',
                                        shape=[vocabulary_length, LEARNED_EMBEDDING_SIZE],
                                        initializer=tf.contrib.layers.xavier_initializer())

tf_message_embs = tf.nn.embedding_lookup(tf_learned_embeddings, tf_message, name='message_embeddings')

with tf.variable_scope('MESSAGE_ENCODER'):
    message_lstm = tf.contrib.rnn.LSTMCell(num_units=RNN_HIDDEN_DIM)
    tf_message_outputs, tf_message_state = tf.nn.dynamic_rnn(message_lstm, tf_message_embs, dtype=tf.float32)

tf_message_final_output_tile = tf.tile(tf.reshape(tf_message_outputs[:, -1, :], [-1, 1, RNN_HIDDEN_DIM]),
                                       [1, MAX_MESSAGE_LENGTH, 1])

with tf.variable_scope('MESSAGE_DECODER'):
    response_lstm = tf.contrib.rnn.LSTMCell(num_units=RNN_HIDDEN_DIM)
tf_response_outputs, tf_response_state = tf.nn.dynamic_rnn(response_lstm, tf_message_final_output_tile,
                                                           dtype=tf.float32,
                                                           initial_state=tf_message_state)

with tf.variable_scope('OUTPUT_PREDICTION'):
    print('Creating output layer...')
    output_weight = tf.get_variable('output_weight',
                                    shape=[RNN_HIDDEN_DIM, vocabulary_length],
                                    initializer=tf.contrib.layers.xavier_initializer())
    output_bias = tf.get_variable('output_bias',
                                  shape=[vocabulary_length],
                                  initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope('tf_message_log_probabilities'):
        tf_response_outputs_reshape = tf.reshape(tf_response_outputs, [-1, RNN_HIDDEN_DIM])
        tf_message_log_probabilities = tf.reshape(tf.matmul(tf_response_outputs_reshape, output_weight) + output_bias,
                                                  [-1, MAX_MESSAGE_LENGTH, vocabulary_length])

    tf_message_probabilities = tf.nn.softmax(tf_message_log_probabilities, name='message_probabilities')

    tf_message_prediction = tf.argmax(tf_message_probabilities, axis=2)
    print('tf_message_prediction shape: %s' % str(tf_message_prediction.get_shape()))

with tf.variable_scope('LOSS'):
    tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_message_log_probabilities,
                                                               labels=tf_message,
                                                               name='word_losses')
    with tf.name_scope('total_loss'):
        tf_total_loss = tf.reduce_sum(tf_losses) / tf.cast(tf_batch_size, tf.float32)

baseline_model_func.create_tensorboard_visualization('chat')

with tf.name_scope("SAVER"):
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf_total_loss)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

if RESTORE_FROM_SAVE:
    print('Restoring from save...')
    baseline_model_func.restore_model_from_save(config.CHAT_MODEL_SAVE_DIR, var_list=tf.trainable_variables(), sess=sess)

num_train_messages = int(num_messages * TRAINING_FRACTION)
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
for index, message_reconstruct in enumerate(val_message_reconstruct[:NUM_EXAMPLES_TO_PRINT]):
    print(message_reconstruct, '\t\t\t|||', ' '.join(messages[num_train_messages + index]))
