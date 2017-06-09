"""Trains and predicts using the baseline LSTM model."""
print('Starting program')
import tensorflow as tf
import squad_dataset_tools as sdt
import config
import baseline_model
import numpy as np
import attention
import os

# CONTROL PANEL
LEARNING_RATE = .0001
NUM_PARAGRAPHS = 50
RNN_HIDDEN_DIM = 800
NUM_EXAMPLES_TO_PRINT = 20
TRAIN_FRAC = 0.8
PREDICT_PROBABILITIES = True
VALIDATE_PROPER_INPUTS = False
RESTORE_FROM_SAVE = False
TRAIN_MODEL_BEFORE_PREDICTION = True
NUM_EPOCHS = 20
PRINT_TRAINING_EXAMPLES = True
PRINT_VALIDATION_EXAMPLES = True
PRINT_ACCURACY_EVERY_N_BATCHES = None
BATCH_SIZE = 20

if not os.path.exists(config.BASELINE_MODEL_SAVE_DIR):
    os.makedirs(config.BASELINE_MODEL_SAVE_DIR)

tf_config = tf.ConfigProto()
tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

print('Loading SQuAD dataset')
paragraphs = sdt.load_squad_dataset_from_file(config.SQUAD_TRAIN_SET)
if NUM_PARAGRAPHS is not None:
    paragraphs = paragraphs[:NUM_PARAGRAPHS]
print('Processing %s paragraphs...' % len(paragraphs))
print('Tokenizing paragraph samples')
tk_paragraphs = sdt.tokenize_paragraphs(paragraphs)
print('Building vocabulary')
vocab_dict = sdt.generate_vocabulary_for_paragraphs(tk_paragraphs).token2id
vocabulary = sdt.invert_dictionary(vocab_dict)  # becomes list of words essentially
vocabulary_size = len(vocab_dict)
print('Len vocabulary: %s' % vocabulary_size)
print('Flatting paragraphs into examples')
examples = sdt.convert_paragraphs_to_flat_format(tk_paragraphs)
print('Converting each example to numpy arrays')
np_questions, np_answers, np_contexts, ids, np_as \
    = sdt.generate_numpy_features_from_squad_examples(examples, vocab_dict,
                                                      answer_indices_from_context=True,
                                                      answer_is_span=False)
np_answer_masks = sdt.compute_answer_mask(np_answers)
print('Maximum index in answers should be less than max context size + 1: %s' % np_answers.max())
num_examples = np_questions.shape[0]
print('Number of examples: %s' % np_questions.shape[0])
contexts = [example[2] for example in examples]
np_context_lengths = np.array([len(context.split()) for context in contexts])
questions = [example[0] for example in examples]
np_question_lengths = np.array([len(question.split()) for question in questions])
print('Average context length: %s' % np.mean(np_context_lengths))
print('Context length deviation: %s' % np.std(np_context_lengths))
print('Max context length: %s' % np.max(np_context_lengths))
num_empty_answers = 0
for i in range(np_answers.shape[0]):
    if np.isclose(np_answers[i], np.zeros([np_answers.shape[1]])).all():
        num_empty_answers += 1
print('Fraction of empty answer vectors (should be close to zero): %s' % (num_empty_answers / num_examples))
print('Loading embeddings for each word in vocabulary')
np_embeddings = sdt.construct_embeddings_for_vocab(vocab_dict)
num_embs = np_embeddings.shape[0]
emb_size = np_embeddings.shape[1]
num_empty_embs = 0
print('Embedding shape: %s' % str(np_embeddings[0, :].shape))
for i in range(num_embs):
    if np.isclose(np_embeddings[i, :], np.zeros([emb_size])).all():
        num_empty_embs += 1
fraction_empty_embs = num_empty_embs / num_embs
print('Fraction of empty embeddings in vocabulary: %s' % fraction_empty_embs)
index_prob_size = config.MAX_CONTEXT_WORDS
num_answers_in_context = 0
for each_example in examples:
    answer = each_example[1]
    context = each_example[2]
    if answer in context:
        num_answers_in_context += 1
print('Fraction of answers found in passages: %s' % (num_answers_in_context / num_examples))

print('Loading embeddings into Tensorflow')
tf_embeddings = tf.Variable(np_embeddings, name='word_embeddings', dtype=tf.float32, trainable=False)
print('Constructing placeholders')
tf_question_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_QUESTION_WORDS), name='question_indices')
tf_question_lengths = tf.placeholder(dtype=tf.int32, shape=(None), name='question_lengths')
tf_context_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_CONTEXT_WORDS), name='context_indices')
tf_context_lengths = tf.placeholder(dtype=tf.int32, shape=(None), name='context_lengths')
tf_answer_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_ANSWER_WORDS), name='answer_indices')
tf_answer_masks = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_ANSWER_WORDS), name='answer_masks')
tf_batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')


tf_question_embs = tf.nn.embedding_lookup(tf_embeddings, tf_question_indices, name='question_embeddings')
tf_context_embs = tf.nn.embedding_lookup(tf_embeddings, tf_context_indices, name='context_embeddings')
print('Question embeddings shape: %s' % str(tf_question_embs.shape))
print('Context embeddings shape: %s' % str(tf_context_embs.shape))

# Model
with tf.name_scope('QUESTION_ENCODER'):
    # _, question_hidden_states = baseline_model.build_gru(RNN_HIDDEN_DIM, tf_batch_size,
    #                                                      [tf_question_embs], config.MAX_QUESTION_WORDS,
    #                                                      gru_scope='QUESTION_RNN')
    question_gru = tf.contrib.rnn.GRUCell(num_units=RNN_HIDDEN_DIM)
    question_outputs, tf_question_state = tf.nn.dynamic_rnn(question_gru, tf_question_embs,
                                                            sequence_length=tf_question_lengths, dtype=tf.float32)
# with tf.name_scope('CONTEXT_ENCODER'):
#     _, context_hidden_states = baseline_model.build_gru(RNN_HIDDEN_DIM, tf_batch_size,
#                                                         [tf_context_embs], config.MAX_CONTEXT_WORDS,
#                                                         gru_scope='CONTEXT_RNN')
#
# with tf.name_scope('CONTEXT_ENCODER_REVERSE'):
#     _, reverse_context_hidden_states = baseline_model.build_gru(RNN_HIDDEN_DIM, tf_batch_size,
#                                                                 [tf_context_embs], config.MAX_CONTEXT_WORDS,
#                                                                 gru_scope='CONTEXT_RNN_REVERSE', reverse=True)
with tf.name_scope('CONTEXT_ENCODER'):


with tf.name_scope('MATCH_GRU'):
    match_gru = tf.contrib.rnn.GRUCell(num_units=RNN_HIDDEN_DIM)
    match_gru_reverse = tf.contrib.rnn.GRUCell(num_units=RNN_HIDDEN_DIM)

    tf_question_state_reshaped = tf.tile(tf.reshape(tf_question_state, (-1, 1, RNN_HIDDEN_DIM)), (1, config.MAX_CONTEXT_WORDS, 1))

    tf_context_inputs = tf.concat([tf_question_state_reshaped, tf_context_embs], axis=2)

    context_outputs, context_output_states = tf.nn.bidirectional_dynamic_rnn(match_gru, match_gru_reverse, tf_context_inputs,
                                                                             sequence_length=tf_context_lengths,
                                                                             dtype=tf.float32)
print('Below: tf_fw_bw_outputs, output_states[0] shapes -')
tf_fw_bw_outputs = tf.concat(context_outputs, axis=2)
print(tf_fw_bw_outputs.get_shape())
print(context_output_states[0].get_shape())

# print('tf_context_hidden_states shape: %s' % str(tf_context_hidden_states.get_shape()))

# tf_weighted_states = attention.attention(tf_fw_bw_outputs, RNN_HIDDEN_DIM)  # attention
# print('Weighted states shape: %s' % str(tf_weighted_states.get_shape()))

# with tf.name_scope("attention"):
#     W_att = tf.Variable(tf.random_uniform([RNN_HIDDEN_DIM * 2, RNN_HIDDEN_DIM * 2], -0.08, 0.08), name='W_att_current')
#     W_question = tf.Variable(tf.random_uniform([RNN_HIDDEN_DIM, RNN_HIDDEN_DIM * 2], -0.08, 0.08), name='W_att_question')
#     # W_att_past = tf.Variable(tf.random_uniform([self.rnn_size, 16], -0.08, 0.08), name='W_att_past')
#     b_att = tf.Variable(tf.zeros([RNN_HIDDEN_DIM * 2]), name='b_att')
#     v_att = tf.Variable(tf.random_uniform([RNN_HIDDEN_DIM * 2], -0.08, 0.08), name='v_att')
#     # rnn_outputs_init = tf.stack([rnn_initial_h, rnn_outputs])
#     rnn_outputs_init_reshaped = tf.reshape(tf_fw_bw_outputs, (-1, RNN_HIDDEN_DIM * 2))
#     question_att = tf.matmul(tf_question_state, W_question)
#     question_att_reshaped = tf.reshape(question_att, [-1, 1, RNN_HIDDEN_DIM * 2])
#     # rnn_outputs_init_reshaped = tf.stack([rnn_initial_h, rnn_outputs_init])
#     att_mat_mul = tf.reshape(tf.matmul(rnn_outputs_init_reshaped, W_att), (-1, config.MAX_CONTEXT_WORDS, RNN_HIDDEN_DIM * 2)) + question_att_reshaped
#     att_add_b = tf.tanh(tf.transpose(tf.transpose(att_mat_mul, perm=[1, 0, 2]) + b_att, perm=[1, 0, 2]))
#     att_v_dot = tf.matmul(tf.reshape(att_add_b, (-1, RNN_HIDDEN_DIM * 2)), tf.expand_dims(v_att, 1))
#     att_raw = tf.reshape(att_v_dot, (-1, config.MAX_CONTEXT_WORDS))
#     att_probs = tf.nn.softmax(att_raw)
#     weighted_states = tf.expand_dims(att_probs, 2) * tf_fw_bw_outputs  # tf.reshape(rnn_outputs, (-1, self.max_len))
#     h_att = tf.reduce_sum(weighted_states, 1)

Hr = tf.concat(context_outputs, axis=2)
# Difference from paper - the empty token goes at the beginning, indicating the index 0 -> ''
Hr_tilda = tf.concat([tf.zeros([tf_batch_size, 1, 2 * RNN_HIDDEN_DIM]), Hr], axis=1)

V = tf.Variable(tf.random_uniform([RNN_HIDDEN_DIM * 2, RNN_HIDDEN_DIM], -0.08, 0.08))
W_a = tf.Variable(tf.random_uniform([RNN_HIDDEN_DIM, RNN_HIDDEN_DIM], -0.08, 0.08))
b_a = tf.Variable(tf.random_uniform([RNN_HIDDEN_DIM], -0.08, 0.08))
v = tf.Variable(tf.random_uniform([RNN_HIDDEN_DIM, 1], -0.08, 0.08))
c = tf.Variable(tf.random_uniform((1, 1), -0.08, 0.08))

answer_gru = tf.contrib.rnn.GRUCell(num_units=RNN_HIDDEN_DIM)
tf_hidden_state = answer_gru.zero_state(tf_batch_size, tf.float32)
B_k_predictions = []
for i in range(config.MAX_ANSWER_WORDS):
    Hr_tilda_V_matmul = tf.reshape(tf.matmul(tf.reshape(Hr_tilda, [-1, RNN_HIDDEN_DIM * 2]), V), [-1, Hr_tilda.shape[1].value, RNN_HIDDEN_DIM])
    F_k = tf.tanh(Hr_tilda_V_matmul + tf.reshape(tf.matmul(tf_hidden_state, W_a) + b_a, [-1, 1, RNN_HIDDEN_DIM]))  # Should broadcast
    F_k_v_matmul = tf.reshape(tf.matmul(tf.reshape(F_k, [-1, RNN_HIDDEN_DIM]), v), [-1, Hr_tilda.shape[1].value, 1])
    B_k = F_k_v_matmul + c
    B_k_predictions.append(tf.reshape(B_k, [-1, 1, Hr_tilda.shape[1].value]))
    tf_answer_input = tf.reshape(tf.matmul(F_k, tf.nn.softmax(B_k), transpose_a=True), [-1, RNN_HIDDEN_DIM])
    with tf.variable_scope('ANSWER_DECODER') as scope:
        if i > 0:
            scope.reuse_variables()
        tf_lstm_output, tf_hidden_state = answer_gru(tf_answer_input, tf_hidden_state)
tf_B_k_predictions = tf.concat(B_k_predictions, axis=1)
# tf_question_context_emb = tf.concat([context_output_states[0], context_output_states[1]], axis=1, name='question_context_emb')
# print('tf_question_context_emb shape: %s' % str(tf_question_context_emb.get_shape()))
#
# if PREDICT_PROBABILITIES:
#     fc_output_size = index_prob_size * 2
# else:
#     fc_output_size = 2
#
# tf_layer1, tf_prediction1_w, tf_prediction1_b = baseline_model.create_dense_layer(tf_question_context_emb, tf_question_context_emb.shape[1].value,
#                                                                                   RNN_HIDDEN_DIM * 2, activation='relu',
#                                                                                   name='FIRST_PREDICTION_LAYER')
#
# tf_start_end_indices, tf_prediction2_w, tf_prediction2_b = baseline_model.create_dense_layer(tf_layer1, RNN_HIDDEN_DIM * 2,
#                                                                                              fc_output_size, activation=None,
#                                                                                              name='SECOND_PREDICTION_LAYER')

# if PREDICT_PROBABILITIES:
#     tf_start_prob = tf_start_end_indices[:, :index_prob_size]
#     tf_end_prob = tf_start_end_indices[:, index_prob_size:]
#     print('tf_start_prob, tf_answer_indices[:, 0]')
#     print(tf_start_prob.get_shape())
#     print(tf_answer_indices[:, 0].get_shape())
#     tf_start_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_start_prob,
#                                                                      labels=tf_answer_indices[:, 0])
#     tf_end_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_end_prob,
#                                                                    labels=tf_answer_indices[:, 1])
#     tf_total_loss = tf.reduce_mean(tf_start_losses + tf_end_losses, axis=0)
#     tf_start_index = tf.argmax(tf_start_prob, axis=1)
#     tf_end_index = tf.argmax(tf_end_prob, axis=1)
#     tf_prediction = tf.stack([tf_start_index, tf_end_index], axis=1)
# else:
#     # Predict real values for indices
#     tf_start_index = tf_start_end_indices[:, 0]
#     tf_end_index = tf_start_end_indices[:, 1]
#     tf_total_loss = tf.nn.l2_loss(tf.cast(tf_answer_indices, tf.float32) - tf_start_end_indices, name='loss')  # squared error
#     tf_total_loss = tf_total_loss / tf.cast(tf_batch_size, tf.float32)
#     tf_prediction = tf.cast(tf.round(tf_start_end_indices), tf.int32)

# vocab_predictions = []
# for each_output in answer_outputs:
#     word_prediction = tf.matmul(each_output, tf_vocab_w) + tf_vocab_b
#     vocab_predictions.append(word_prediction)
#
tf_prediction = tf.argmax(tf_B_k_predictions, axis=2)
#
# print('Model output prediction indices shape: %s' % str(tf_predictions_indices.get_shape()))
# print('RNN output shape: %s' % str(answer_outputs[0].get_shape()))
# print('Labels indices for loss function: %s' % str(tf_answer_indices[:, 0].get_shape()))
# print('Prediction distribution shape: %s' % str(vocab_predictions[0].get_shape()))

# Calculate loss per each
tf_total_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_B_k_predictions, labels=tf_answer_indices)
tf_total_loss = tf.reduce_mean(tf.multiply(tf_total_losses, tf.cast(tf_answer_masks, tf.float32)))
# for i, each_prediction in enumerate(B_k_predictions):
#     tf_output_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=each_prediction, labels=tf_answer_indices[:, i])
#     tf_total_loss += tf.reduce_mean(tf_output_loss, axis=0)
#
# tf_total_loss = tf.reduce_mean(tf_total_losses, axis=0)
# print('Shape of tf_total_loss: %s' % str(tf_total_loss.get_shape()))
# Visualize
baseline_model.create_tensorboard_visualization('cic')

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf_total_loss)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession(config=tf_config)
assert tf_embeddings not in tf.trainable_variables()
with tf.name_scope("SAVER"):
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
sess.run(init)

if VALIDATE_PROPER_INPUTS:
    print('Validating proper inputs')
    # Validates that embedding lookups for the first 1000 contexts and questions
    # look up the correct embedding for each index, and that the embedding is
    # the same as that stored in the embedding table and in the spacy nlp object.
    sample_size = 1000
    if sample_size > num_examples:
        sample_size = num_examples
    fd={tf_context_indices: np_contexts[:sample_size, :]}
    np_sample_context_embs = tf_context_embs.eval(fd)
    for i in range(sample_size):
        context_tokens = contexts[i].split()
        for j in range(np_sample_context_embs.shape[1]):
            if j < len(context_tokens):
                word = context_tokens[j]
                word_vector = sdt.nlp(word).vector
                word_index = vocab_dict[word]
                stored_vector = np_embeddings[word_index, :]
                if not np.isclose(word_vector, np.zeros([config.SPACY_GLOVE_EMB_SIZE])).all():
                    assert np.isclose(np_sample_context_embs[i, j, :], word_vector).all()
                    assert np.isclose(np_sample_context_embs[i, j, :], stored_vector).all()
    fd={tf_question_indices: np_questions[:sample_size, :]}
    np_sample_question_embs = tf_question_embs.eval(fd)
    for i in range(sample_size):
        question_tokens = questions[i].split()
        for j in range(np_sample_question_embs.shape[1]):
            if j < len(question_tokens):
                word = question_tokens[j]
                word_index = vocab_dict[word]
                stored_vector = np_embeddings[word_index, :]
                word_vector = sdt.nlp(word).vector
                if not np.isclose(word_vector, np.zeros([config.SPACY_GLOVE_EMB_SIZE])).all():
                    assert np.isclose(np_sample_question_embs[i, j, :], word_vector).all()
                    assert np.isclose(np_sample_question_embs[i, j, :], stored_vector).all()
    print('Inputs validated')

num_batches = int(num_examples * TRAIN_FRAC / BATCH_SIZE)
num_train_examples = BATCH_SIZE * num_batches
val_index_start = BATCH_SIZE * num_batches

if RESTORE_FROM_SAVE:
    print('Restoring from save...')
    baseline_model.restore_model_from_save(config.BASELINE_MODEL_SAVE_DIR, var_list=tf.trainable_variables(), sess=sess)

if TRAIN_MODEL_BEFORE_PREDICTION:
    print('Training model...')
    all_train_predictions = []
    for epoch in range(NUM_EPOCHS):
        print('Epoch: %s' % epoch)
        losses = []
        accuracies = []
        frac_zeros =[]
        for i in range(num_batches):
            np_question_batch = np_questions[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE, :]
            np_answer_batch = np_answers[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE, :]
            np_answer_mask_batch = np_answer_masks[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE, :]
            np_context_batch = np_contexts[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE, :]
            np_context_length_batch = np_context_lengths[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]
            np_question_length_batch = np_question_lengths[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]
            np_batch_predictions, np_loss, _ = sess.run([tf_prediction, tf_total_loss, train_op],
                                                        feed_dict={tf_question_indices: np_question_batch,
                                                                   tf_question_lengths: np_question_length_batch,
                                                                   tf_answer_indices: np_answer_batch,
                                                                   tf_answer_masks: np_answer_mask_batch,
                                                                   tf_context_indices: np_context_batch,
                                                                   tf_context_lengths: np_context_length_batch,
                                                                   tf_batch_size: BATCH_SIZE})
            accuracy = sdt.compute_multi_label_accuracy(np_answer_batch, np_batch_predictions)
            frac_zero = sdt.compute_multi_label_accuracy(np_batch_predictions, np.zeros([BATCH_SIZE, config.MAX_ANSWER_WORDS]))
            accuracies.append(accuracy)
            frac_zeros.append(frac_zero)
            if PRINT_ACCURACY_EVERY_N_BATCHES is not None and i % PRINT_ACCURACY_EVERY_N_BATCHES == 0:
                print('Batch TRAIN EM Score: %s' % np.mean(accuracies))
            losses.append(np_loss)
            if epoch == NUM_EPOCHS - 1:  # last epoch
                all_train_predictions.append(np_batch_predictions)
        epoch_loss = np.mean(losses)
        epoch_accuracy = np.mean(accuracies)
        epoch_frac_zero = np.mean(frac_zeros)
        print('Epoch loss: %s' % epoch_loss)
        print('Epoch TRAIN EM Score: %s' % epoch_accuracy)
        print('Epoch fraction of zero vector answers: %s' % epoch_frac_zero)
        print('Saving model %s' % epoch)
        saver.save(sess, config.BASELINE_MODEL_SAVE_DIR,
                   global_step=epoch)  # Save model after every epoch

    np_train_predictions = np.concatenate(all_train_predictions, axis=0)
    # total_train_accuracy = sdt.compute_multi_label_accuracy(np_train_predictions, np_answers[:val_index_start, :])
    # print('Final TRAIN EM Score: %s' % total_train_accuracy)

    if PRINT_TRAINING_EXAMPLES:
        # Print training examples
        predictions = sdt.convert_numpy_array_answers_to_strings(np_train_predictions[-NUM_EXAMPLES_TO_PRINT:, :],
                                                                 contexts[:NUM_EXAMPLES_TO_PRINT],
                                                                 answer_is_span=False)
        for i, each_prediction in enumerate(predictions):
            print('Prediction: %s' % each_prediction)
            print('Answer: %s' % examples[i][1])
            print('Answer array: %s' % np_answers[i, :])
            print('Prediction array: %s' % np_train_predictions[i, :])
            print('Question: %s' % examples[i][0])
            print('Len context: %s' % len(contexts[i]))
            print()

print('\n######################################\n')
print('Predicting...')
# Must generate validation predictions in batches to avoid OOM error
num_val_examples = num_examples - val_index_start
num_val_batches = int(num_val_examples / BATCH_SIZE) + 1  # +1 to include remainder examples
all_val_predictions = []
for batch_index in range(num_val_batches):
    current_start_index = BATCH_SIZE * batch_index
    if current_start_index + BATCH_SIZE >= num_val_examples:
        effective_batch_size = num_val_examples - current_start_index
    else:
        effective_batch_size = BATCH_SIZE
    current_end_index = current_start_index + effective_batch_size
    np_batch_val_predictions = sess.run(tf_prediction, feed_dict={tf_question_indices: np_questions[current_start_index:current_end_index, :],
                                                            tf_question_lengths: np_question_lengths[current_start_index: current_end_index],
                                                            tf_answer_indices: np_answers[current_start_index:current_end_index, :],
                                                            tf_answer_masks: np_answer_masks[current_start_index:current_end_index, :],
                                                            tf_context_indices: np_contexts[current_start_index:current_end_index, :],
                                                            tf_context_lengths: np_context_lengths[current_start_index: current_end_index],
                                                            tf_batch_size: effective_batch_size})
    all_val_predictions.append(np_batch_val_predictions)
np_val_predictions = np.concatenate(all_val_predictions, axis=0)
if PRINT_VALIDATION_EXAMPLES:
    # Print validation examples
    predictions = sdt.convert_numpy_array_answers_to_strings(np_val_predictions[:NUM_EXAMPLES_TO_PRINT, :],
                                                             contexts[val_index_start:val_index_start+NUM_EXAMPLES_TO_PRINT],
                                                             answer_is_span=False)
    for i, each_prediction in enumerate(predictions):
        print('Prediction: %s' % each_prediction)
        print('Answer: %s' % examples[val_index_start + i][1])
        print('Answer array: %s' % np_answers[val_index_start + i, :])
        print('Prediction array: %s' % np_val_predictions[i, :])
        print('Question: %s' % examples[val_index_start + i][0])
        print('Len context: %s' % len(contexts[val_index_start + i]))
        print('Context: %s' % contexts[val_index_start + i])
        print()

val_accuracy = sdt.compute_multi_label_accuracy(np_val_predictions, np_answers[val_index_start:])

print('VAL EM Score: %s' % val_accuracy)

print('Prediction shape: %s' % str(np_val_predictions.shape))



















