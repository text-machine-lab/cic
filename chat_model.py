"""Model that uses the Cornell Movie Dialogs Corpus to conduct conversations. Here a conversation is a dialogue between
two characters represented by a sequence of messages."""
import config
import gensim
import spacy
import numpy as np
import chat_model_func
import squad_dataset_tools as sdt
import tensorflow as tf
import random

# CONFIGURATION ########################################################################################################

LEARNING_RATE = .0005
NUM_CONVERSATIONS = 100
NUM_EXAMPLES_TO_PRINT = 20
MAX_MESSAGE_LENGTH = 20
LEARNED_EMBEDDING_SIZE = 100
RNN_HIDDEN_DIM = 800
TRAIN_FRACTION = 0.8
BATCH_SIZE = 1
NUM_EPOCHS = 40
REVERSE_INPUT_MESSAGE = True
SHUFFLE_EXAMPLES = True
STOP_TOKEN = '<STOP>'

# PRE-PROCESSING #######################################################################################################

DELIMITER = ' +++$+++ '
movie_lines_file = open(config.CORNELL_MOVIE_LINES_FILE, 'rb')
movie_conversations_file = open(config.CORNELL_MOVIE_CONVERSATIONS_FILE, 'r')
nlp = spacy.load('en')

print('Processing conversations...')
line_index = 0
conversations = []
id_to_message = {}
for each_conversation_line in movie_conversations_file:
    if line_index < NUM_CONVERSATIONS:
        conversation_data = each_conversation_line.split(DELIMITER)
        first_character_id = conversation_data[0]
        second_character_id = conversation_data[1]
        movie_id = conversation_data[2]
        message_ids = conversation_data[3][2:-3].split("', '")
        for each_message_id in message_ids:
            id_to_message[each_message_id] = None
        conversations.append([first_character_id, second_character_id, movie_id, message_ids])
    line_index += 1
print('Number of valid conversations: %s' % len(conversations))

print('Processing messages...')
num_decode_errors = 0
for message_line in movie_lines_file:
    try:
        message_data = message_line.decode('utf-8').split(DELIMITER)
        message_id = message_data[0]
        if message_id in id_to_message:
            character_id = message_data[1]
            movie_id = message_data[2]
            character_name = message_data[3]
            message = message_data[4][:-1]
            tk_message = nlp.tokenizer(message.lower())
            tk_tokens = [str(token) for token in tk_message if str(token) != ' '] + [STOP_TOKEN]
            if len(tk_tokens) <= MAX_MESSAGE_LENGTH:
                id_to_message[message_id] = [character_id, movie_id, character_name, message, tk_tokens]
    except UnicodeDecodeError:
        num_decode_errors += 1
print('Number of decoding errors: %s' % num_decode_errors)
num_messages = len(id_to_message)
print('Number of messages: %s' % num_messages)

num_empty_messages = 0
message_lengths = []
for key in id_to_message:
    if id_to_message[key] is None:
        num_empty_messages += 1
    else:
        message_lengths.append(len(id_to_message[key][-1]))
print('Number of messages not found: %s' % num_empty_messages)
print('Average message length: %s' % np.mean(message_lengths))
print('Message length std: %s' % np.std(message_lengths))
print('Message max length: %s' % np.max(message_lengths))

documents = []
for key in id_to_message:
    each_message_data = id_to_message[key]
    if each_message_data is not None:
        documents.append(each_message_data[-1])
dictionary = gensim.corpora.Dictionary([['']], prune_at=None)
dictionary.add_documents(documents, prune_at=None)
vocab_dict = dictionary.token2id
vocabulary = sdt.invert_dictionary(vocab_dict)
vocabulary_length = len(vocab_dict)
print('Vocabulary size: %s' % vocabulary_length)

print('Creating examples...')
examples = []
for each_conversation in conversations:
    conversation_message_ids = each_conversation[-1]
    for message_index in range(1, len(conversation_message_ids)):
        first_message_id = conversation_message_ids[message_index - 1]
        second_message_id = conversation_message_ids[message_index]
        if id_to_message[first_message_id] is not None and id_to_message[second_message_id] is not None:
            each_message = id_to_message[first_message_id][-1]
            each_response = id_to_message[second_message_id][-1]
            examples.append([each_message, each_response])
num_examples = len(examples)
print('Example example: %s' % str(examples[0]))
print('Number of examples: %s' % num_examples)

if SHUFFLE_EXAMPLES:
    random.shuffle(examples)

print('Constructing input numpy arrays...')
np_message, np_response = chat_model_func.construct_numpy_from_examples(examples, vocab_dict, MAX_MESSAGE_LENGTH)
message_reconstruct = sdt.convert_numpy_array_to_strings(np_message, vocabulary)
response_reconstruct = sdt.convert_numpy_array_to_strings(np_response, vocabulary)
for i in range(len(examples)):
    each_message = ' '.join(examples[i][0])
    each_response = ' '.join(examples[i][1])

    if len(examples[i][0]) <= MAX_MESSAGE_LENGTH:
        assert each_message == message_reconstruct[i]
    if len(examples[i][1]) <= MAX_MESSAGE_LENGTH:
        assert each_response == response_reconstruct[i]

if REVERSE_INPUT_MESSAGE:
    np_message = np.flip(np_message, axis=1)

# GRAPH CREATION #######################################################################################################

print('Constructing model...')
tf_message = tf.placeholder(dtype=tf.int32, shape=[None, MAX_MESSAGE_LENGTH], name='input_message')
tf_response = tf.placeholder(dtype=tf.int32, shape=[None, MAX_MESSAGE_LENGTH], name='output_response')
tf_response_mask = tf.not_equal(tf_response, tf.constant(0))
print('tf_response_mask shape: %s' % str(tf_response_mask.get_shape()))

tf_learned_embeddings = tf.get_variable('learned_embeddings',
                                        shape=[vocabulary_length, LEARNED_EMBEDDING_SIZE],
                                        initializer=tf.contrib.layers.xavier_initializer())

tf_message_embs = tf.nn.embedding_lookup(tf_learned_embeddings, tf_message, name='message_embeddings')

with tf.variable_scope('MESSAGE_ENCODER'):
    message_gru = tf.contrib.rnn.GRUCell(num_units=RNN_HIDDEN_DIM)
    tf_message_outputs, tf_message_state = tf.nn.dynamic_rnn(message_gru, tf_message_embs, dtype=tf.float32)

tf_message_state_tile = tf.tile(tf.reshape(tf_message_state, [-1, 1, RNN_HIDDEN_DIM]), [1, MAX_MESSAGE_LENGTH, 1])

with tf.variable_scope('RESPONSE_DECODER'):
    response_gru = tf.contrib.rnn.GRUCell(num_units=RNN_HIDDEN_DIM)
    tf_response_outputs, tf_response_state = tf.nn.dynamic_rnn(response_gru, tf_message_state_tile, dtype=tf.float32)

with tf.variable_scope('OUTPUT_PREDICTION'):
    W_v = tf.get_variable('output_weight',
                          shape=[RNN_HIDDEN_DIM, vocabulary_length],
                          initializer=tf.contrib.layers.xavier_initializer())
    W_b = tf.get_variable('output_bias',
                          shape=[vocabulary_length],
                          initializer=tf.contrib.layers.xavier_initializer())

    # tf_vocab_compare_output = tf.reshape(tf.matmul(tf.reshape(tf_response_outputs, [-1, RNN_HIDDEN_DIM]), W_v) + W_b,
    #                                      [-1, LEARNED_EMBEDDING_SIZE])
    #
    # tf_response_log_probabilities = tf.reshape(tf.matmul(tf_vocab_compare_output, tf_learned_embeddings, transpose_b=True),
    #                                            [-1, MAX_MESSAGE_LENGTH, vocabulary_length])

    tf_response_log_probabilities = tf.reshape(tf.matmul(tf.reshape(tf_response_outputs, [-1, RNN_HIDDEN_DIM]), W_v) + W_b,
                                               [-1, MAX_MESSAGE_LENGTH, vocabulary_length])

    tf_response_probabilities = tf.nn.softmax(tf_response_log_probabilities)

    tf_response_prediction = tf.argmax(tf_response_probabilities, axis=2)
    print('tf_response_prediction shape: %s' % str(tf_response_prediction.get_shape()))

with tf.variable_scope('LOSS'):
    tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_response_log_probabilities,
                                                               labels=tf_response)
    tf_total_loss = tf.reduce_mean(tf.multiply(tf_losses, tf.cast(tf_response_mask, tf.float32)))

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf_total_loss)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# TRAINING #############################################################################################################

num_batches = int(num_examples * TRAIN_FRACTION / BATCH_SIZE)
num_train_examples = num_batches * BATCH_SIZE
np_train_message = np_message[:num_train_examples, :]
np_train_response = np_response[:num_train_examples, :]
train_examples = examples[:num_train_examples]

for epoch in range(NUM_EPOCHS):
    print('Epoch: %s' % epoch)
    batch_losses = []
    for batch_index in range(num_batches):
        np_batch_message = np_train_message[batch_index * BATCH_SIZE:batch_index * BATCH_SIZE + BATCH_SIZE, :]
        np_batch_response = np_train_response[batch_index * BATCH_SIZE:batch_index * BATCH_SIZE + BATCH_SIZE, :]
        batch_loss, batch_response_predictions, _, batch_mask = sess.run([tf_total_loss, tf_response_prediction, train_op, tf_response_mask],
                                                             feed_dict={tf_message: np_batch_message,
                                                                        tf_response: np_batch_response})
        batch_losses.append(batch_loss)
    print('Epoch loss: %s' % np.mean(batch_losses))

np_train_predictions = sess.run(tf_response_prediction, feed_dict={tf_message: np_train_message})

train_predictions = sdt.convert_numpy_array_to_strings(np_train_predictions, vocabulary, stop_token=STOP_TOKEN)

print('Printing training examples...')
for i, each_prediction in enumerate(train_predictions):
    if i < NUM_EXAMPLES_TO_PRINT:
        print('Message: %s' % (' '.join(train_examples[i][0])))
        print('Response: %s' % each_prediction)
        print('Label array: %s' % np_train_response[i, :])
        print('Prediction array: %s' % np_train_predictions[i, :])

# PREDICTION ###########################################################################################################
np_val_message = np_message[num_train_examples:, :]
np_val_response = np_response[num_train_examples:, :]
np_val_predictions = sess.run(tf_response_prediction, feed_dict={tf_message: np_val_message})
val_examples = examples[num_train_examples:]

val_predictions = sdt.convert_numpy_array_to_strings(np_val_predictions, vocabulary, stop_token=STOP_TOKEN)
print('\nPrinting validation examples...')
for i, each_prediction in enumerate(val_predictions):
    if i < NUM_EXAMPLES_TO_PRINT:
        print('Message: %s' % (' '.join(val_examples[i][1])))
        print('Response: %s' % each_prediction)
        print('Response array: %s' % np_val_response[i, :])
        print('Prediction array: %s' % np_val_predictions[i, :])

# CHAT #################################################################################################################
# Cannot enter words that are out of vocabulary
print('Chat with the chat bot! Enter a message:')
while True:
    chat_message = input('You: ')
    tk_chat_message = nlp.tokenizer(chat_message.lower())
    tk_chat_tokens = [str(token) for token in tk_chat_message if str(token) != ' ' and str(token) in vocab_dict] + [STOP_TOKEN]
    print(tk_chat_tokens)
    np_chat_message = chat_model_func.construct_numpy_from_messages([tk_chat_tokens], vocab_dict, MAX_MESSAGE_LENGTH)
    np_chat_response = sess.run(tf_response_prediction, feed_dict={tf_message: np_chat_message})
    response = sdt.convert_numpy_array_to_strings(np_chat_response, vocabulary, stop_token=STOP_TOKEN)
    print('Bot: %s' % response[0])



