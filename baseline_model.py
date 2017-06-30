"""Trains and predicts a model using a Match-LSTM and Answer Pointer. Design based on this paper:
https://arxiv.org/pdf/1608.07905.pdf"""
import tensorflow as tf
import squad_dataset_tools as sdt
import config
import baseline_model_func
import numpy as np
import os
import random
import pickle as pkl
import sys
import json

sdt.initialize_nlp()

# CONTROL PANEL ########################################################################################################

LEARNING_RATE = .001
NUM_PARAGRAPHS = None
RNN_HIDDEN_DIM = 200
NUM_EXAMPLES_TO_PRINT = 40
TRAIN_FRAC = 0.8
VALIDATE_PROPER_INPUTS = True
RESTORE_FROM_SAVE = False
TRAIN_MODEL_BEFORE_PREDICTION = True
PREDICT_ON_TRAINING_EXAMPLES = False  # Predict on all training examples after training
NUM_EPOCHS = 50
PRINT_TRAINING_EXAMPLES = True
PRINT_VALIDATION_EXAMPLES = True
PRINT_ACCURACY_EVERY_N_BATCHES = None
BATCH_SIZE = 20
STOP_TOKEN_REWARD = 2
TURN_OFF_TF_LOGGING = True
USE_SPACY_NOT_GLOVE = True  # Use Spacy GloVe embeddings or Twitter Glove embeddings
SHUFFLE_EXAMPLES = True
SIMILARITY_LOSS_CONST = 0
SAVE_VALIDATION_PREDICTIONS = True
PRODUCE_OUTPUT_PREDICTIONS_FILE = False
REMOVE_EXAMPLES_GREATER_THAN_MAX_LENGTH = False  # Creates an unrealistic dataset with no cropping
REMOVE_EXAMPLES_WITH_MIN_FRAC_EMPTY_EMBEDDINGS = 0.0

# SUBMISSION ###########################################################################################################

# If True, we are in submit mode. Model should take in a json file, make predictions, and produce an output file.
SUBMISSION_MODE = False
if len(sys.argv) > 1:
    SUBMISSION_MODE = True
    print('Entering submission mode')
    config.SQUAD_TRAIN_SET = sys.argv[1]
    TRAIN_MODEL_BEFORE_PREDICTION = False
    NUM_PARAGRAPHS = None
    TRAIN_FRAC = 0
    RESTORE_FROM_SAVE = True
    PRINT_TRAINING_EXAMPLES = False
    PRINT_VALIDATION_EXAMPLES = False
    PRINT_ACCURACY_EVERY_N_BATCHES = None
    SAVE_VALIDATION_PREDICTIONS = False
    VALIDATE_PROPER_INPUTS = False
    PREDICT_ON_TRAINING_EXAMPLES = False
    PRODUCE_OUTPUT_PREDICTIONS_FILE = True

if len(sys.argv) > 2:
    config.SUBMISSION_PREDICTIONS_FILE = sys.argv[2]

if len(sys.argv) > 3:
    config.BASELINE_MODEL_SAVE_DIR = sys.argv[3]

# PRE-PROCESSING #######################################################################################################

if not USE_SPACY_NOT_GLOVE:
    config.GLOVE_EMB_SIZE = 200

if not os.path.exists(config.BASELINE_MODEL_SAVE_DIR):
    os.makedirs(config.BASELINE_MODEL_SAVE_DIR)

if TURN_OFF_TF_LOGGING:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#tf_config = tf.ConfigProto()
# tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

print('Loading SQuAD dataset')
paragraphs = sdt.load_squad_dataset_from_file(config.SQUAD_TRAIN_SET)
#paragraphs = [paragraph for paragraph in paragraphs if len(paragraph['context'].split()) <= config.MAX_CONTEXT_WORDS]
if NUM_PARAGRAPHS is not None:
    paragraphs = paragraphs[:NUM_PARAGRAPHS]

print('Processing %s paragraphs...' % len(paragraphs))
print('Tokenizing paragraph samples')
tk_paragraphs = sdt.tokenize_paragraphs(paragraphs)
print('Removing excess spaces')
clean_paragraphs = sdt.remove_excess_spaces_from_paragraphs(tk_paragraphs)

print('Building vocabulary')
vocab_dict = sdt.generate_vocabulary_for_paragraphs(clean_paragraphs).token2id
vocabulary = sdt.invert_dictionary(vocab_dict)  # becomes list of words essentially
vocabulary_size = len(vocab_dict)
print('Len vocabulary: %s' % vocabulary_size)

print('Loading embeddings for each word in vocabulary')
np_embeddings = sdt.construct_embeddings_for_vocab(vocab_dict, use_spacy_not_glove=USE_SPACY_NOT_GLOVE)
num_embs = np_embeddings.shape[0]
emb_size = np_embeddings.shape[1]
num_empty_embs = 0
print('Embedding shape: %s' % str(np_embeddings[0, :].shape))
for i in range(num_embs):
    if np.isclose(np_embeddings[i, :], np.zeros([emb_size])).all():
        num_empty_embs += 1
        # print(vocabulary[i])
fraction_empty_embs = num_empty_embs / num_embs
print('Fraction of empty embeddings in vocabulary: %s' % fraction_empty_embs)

print('Flattening paragraphs into examples')
examples = sdt.convert_paragraphs_to_flat_format(clean_paragraphs)

if REMOVE_EXAMPLES_GREATER_THAN_MAX_LENGTH:
    examples = [example for example in examples
                if len(example[0].split()) < config.MAX_QUESTION_WORDS
                and len(example[1].split()) < config.MAX_ANSWER_WORDS
                and len(example[2].split()) < config.MAX_CONTEXT_WORDS]

dense_examples = []
if REMOVE_EXAMPLES_WITH_MIN_FRAC_EMPTY_EMBEDDINGS != 0.0:
    for example in examples:
        question = example[0].split()
        context = example[1].split()
        num_empty_question_embeddings = 0
        for each_token in question:
            token_index = vocab_dict[each_token]
            if not np_embeddings[token_index, :].any():
                num_empty_question_embeddings += 1
        frac_empty_question_embeddings = num_empty_question_embeddings / len(question)
        if frac_empty_question_embeddings < REMOVE_EXAMPLES_WITH_MIN_FRAC_EMPTY_EMBEDDINGS:
            num_empty_context_embeddings = 0
            for each_token in context:
                token_index = vocab_dict[each_token]
                if not np_embeddings[token_index, :].any():
                    num_empty_context_embeddings += 1
            frac_empty_context_embeddings = num_empty_context_embeddings / len(context)
            if frac_empty_context_embeddings < REMOVE_EXAMPLES_WITH_MIN_FRAC_EMPTY_EMBEDDINGS:
                dense_examples.append(example)
    examples = dense_examples

if SHUFFLE_EXAMPLES:
    random.shuffle(examples)
print('Converting each example to numpy arrays')
np_questions, np_answers, np_contexts, ids, np_as \
    = sdt.generate_numpy_features_from_squad_examples(examples, vocab_dict,
                                                      answer_indices_from_context=True,
                                                      answer_is_span=False)
np_answer_masks = sdt.compute_answer_mask(np_answers, stop_token=True, zero_weight=STOP_TOKEN_REWARD)
print('Mean answer mask value: %s' % np.mean(np_answer_masks))
print('Maximum index in answers should be less than max context size + 1: %s' % np_answers.max())
num_examples = np_questions.shape[0]
print('Number of examples: %s' % np_questions.shape[0])

# Check fraction of answers that can be detokenized
num_detokenized_answers = 0
for i in range(len(paragraphs)):
    for j in range(len(paragraphs[i]['qas'])):
        for k in range(len(paragraphs[i]['qas'][j]['answers'])):
            text = paragraphs[i]['qas'][j]['answers'][k]['text']
            normalized_text = sdt.normalize_answer(text)

            tk_text = clean_paragraphs[i]['qas'][j]['answers'][k]['text']
            detk_text = sdt.detokenize_string(tk_text)
            normalized_detk_text = sdt.normalize_answer(detk_text)

            if normalized_text == normalized_detk_text:
                num_detokenized_answers += 1
            elif i < 10:
                print('Normalized text: %s' % normalized_text)
                print('Tokenized text: %s' % tk_text)
                print('Detokenized text: %s' % detk_text)
                print('Normalized detokenized text: %s' % normalized_detk_text)
                print()

print('Fraction detokenized answers: %s' % (num_detokenized_answers / len(examples)))

contexts = [example[2] for example in examples]
np_context_lengths = np.array([len(context.split()) for context in contexts])
questions = [example[0] for example in examples]
np_question_lengths = np.array([len(question.split()) for question in questions])
answers = [example[1] for example in examples]
np_answer_lengths = np.array([len(answer.split()) for answer in answers])
print('Average context length: %s' % np.mean(np_context_lengths))
print('Context length deviation: %s' % np.std(np_context_lengths))
print('Max context length: %s' % np.max(np_context_lengths))
print('Average question length: %s' % np.mean(np_question_lengths))
print('Question length deviation: %s' % np.std(np_question_lengths))
print('Max question length: %s' % np.max(np_question_lengths))
print('Average answer length: %s' % np.mean(np_answer_lengths))
print('Answer length deviation: %s' % np.std(np_answer_lengths))
print('Max answer length: %s' % np.max(np_answer_lengths))

if VALIDATE_PROPER_INPUTS:
    print('Validating inputs...')
    assert np_contexts.shape[0] == num_examples
    assert np_contexts.shape[0] == np_answers.shape[0]
    assert np_answers.shape[0] == np_questions.shape[0]
    assert np_questions.shape[0] == np_question_lengths.shape[0]
    assert np_question_lengths.shape[0] == np_answer_lengths.shape[0]
    assert np_answer_lengths.shape[0] == np_context_lengths.shape[0]
    assert np_context_lengths.shape[0] == np_answer_masks.shape[0]
    assert np_contexts.shape[1] == config.MAX_CONTEXT_WORDS
    assert np_answers.shape[1] == config.MAX_ANSWER_WORDS
    assert np_questions.shape[1] == config.MAX_QUESTION_WORDS
    reconstructed_contexts = sdt.convert_numpy_array_to_strings(np_contexts, vocabulary)
    reconstructed_answers = sdt.convert_numpy_array_answers_to_strings(np_answers, contexts)
    reconstructed_questions = sdt.convert_numpy_array_to_strings(np_questions, vocabulary)
    num_corrupted_answers = 0
    for i in range(num_examples):
        assert (reconstructed_contexts[i] == contexts[i] or len(contexts[i].split()) > config.MAX_CONTEXT_WORDS)
        assert (reconstructed_questions[i] == questions[i] or len(questions[i].split()) > config.MAX_QUESTION_WORDS)
        if not (reconstructed_answers[i] == answers[i]
                or len(answers[i].split()) > config.MAX_ANSWER_WORDS
                or len(contexts[i].split()) > config.MAX_CONTEXT_WORDS):
            num_corrupted_answers += 1
            # print('Answer: %s' % answers[i])
            # print('Answer reconstruct: %s' % reconstructed_answers[i])
            # print('Answer vector: %s' % np_answers[i, :])
            # print('Example #: %s' % i)
            # print('Context: %s' % contexts[i])
            # print('Len context: %s' % len(contexts[i].split()))
    print('Fraction of corrupted answers: %s' % (num_corrupted_answers / num_examples))
    print('Validated numpy arrays for contexts and questions')

num_empty_answers = 0
for i in range(np_answers.shape[0]):
    if np.isclose(np_answers[i], np.zeros([np_answers.shape[1]])).all():
        num_empty_answers += 1
print('Fraction of empty answer vectors (should be close to zero): %s' % (num_empty_answers / num_examples))

index_prob_size = config.MAX_CONTEXT_WORDS
num_answers_in_context = 0
for each_example in examples:
    answer = each_example[1]
    context = each_example[2]
    if answer in context:
        num_answers_in_context += 1
print('Fraction of answers found in passages: %s' % (num_answers_in_context / num_examples))

# GRAPH CREATION #######################################################################################################

print('Loading embeddings into Tensorflow')
tf_embeddings = tf.Variable(np_embeddings, name='word_embeddings', dtype=tf.float32, trainable=False)
print('Constructing placeholders')

with tf.name_scope('PLACEHOLDERS'):
    tf_question_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_QUESTION_WORDS), name='question_indices')
    tf_question_lengths = tf.placeholder(dtype=tf.int32, shape=(None), name='question_lengths')
    tf_context_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_CONTEXT_WORDS), name='context_indices')
    tf_context_lengths = tf.placeholder(dtype=tf.int32, shape=(None), name='context_lengths')
    tf_answer_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_ANSWER_WORDS), name='answer_indices')
    tf_answer_masks = tf.placeholder(dtype=tf.float32, shape=(None, config.MAX_ANSWER_WORDS), name='answer_masks')
    tf_batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')

with tf.name_scope('INPUT_EMBEDDINGS'):
    tf_question_embs = tf.nn.embedding_lookup(tf_embeddings, tf_question_indices, name='question_embeddings')
    tf_context_embs = tf.nn.embedding_lookup(tf_embeddings, tf_context_indices, name='context_embeddings')

# Correct so far...

print('Question embeddings shape: %s' % str(tf_question_embs.shape))
print('Context embeddings shape: %s' % str(tf_context_embs.shape))

# Removed sequence lengths from question and context encoders

# Model
with tf.variable_scope('QUESTION_ENCODER'):
    question_lstm = tf.contrib.rnn.LSTMCell(num_units=RNN_HIDDEN_DIM)
    tf_question_outputs, tf_question_state = tf.nn.dynamic_rnn(question_lstm, tf_question_embs,
                                                               sequence_length=None, dtype=tf.float32)

# tf_question_state_reshape = tf.reshape(tf_question_state, [-1, 1, RNN_HIDDEN_DIM])
# tf_question_state_tile = tf.tile(tf_question_state_reshape, [1, config.MAX_CONTEXT_WORDS, 1])
# tf_context_encoder_input = tf.concat([tf_context_embs, tf_question_state_tile], axis=2)
# assert tf_context_encoder_input.shape[1].value == config.MAX_CONTEXT_WORDS
# assert tf_context_encoder_input.shape[2].value == RNN_HIDDEN_DIM + config.GLOVE_EMB_SIZE

with tf.variable_scope('CONTEXT_ENCODER'):
    context_lstm = tf.contrib.rnn.LSTMCell(num_units=RNN_HIDDEN_DIM)
    tf_context_outputs, tf_context_state = tf.nn.dynamic_rnn(context_lstm, tf_context_embs,
                                                             sequence_length=None, dtype=tf.float32)

# Pretty sure this is correct so far...

with tf.variable_scope('MATCH_GRU'):
    with tf.variable_scope('FORWARD'):
        Hr_forward = baseline_model_func.match_gru(tf_question_outputs, tf_context_outputs, tf_batch_size, RNN_HIDDEN_DIM)
    with tf.variable_scope('BACKWARD'):
        Hr_backward = baseline_model_func.match_gru(tf_question_outputs, tf.reverse(tf_context_outputs, [1]), tf_batch_size, RNN_HIDDEN_DIM)
    Hr = tf.concat([Hr_forward, tf.reverse(Hr_backward, [1])], axis=2)

    Hr_tilda = tf.concat([tf.zeros([tf_batch_size, 1, RNN_HIDDEN_DIM * 2]), Hr], axis=1, name='Hr_tilda')

with tf.name_scope('OUTPUT'):
    tf_log_probabilities, all_hidden_states = baseline_model_func.pointer_net(Hr_tilda, tf_batch_size, RNN_HIDDEN_DIM)

    tf_probabilities = tf.nn.softmax(tf_log_probabilities)

    tf_predictions = tf.argmax(tf_probabilities, axis=2, name='predictions')

# Calculate loss per each
with tf.name_scope('LOSS'):
    time_step_losses = []
    for time_step in range(config.MAX_ANSWER_WORDS):
        time_step_losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_log_probabilities[:, time_step, :],
                                                                               labels=tf_answer_indices[:, time_step]))
    tf_total_losses = tf.stack(time_step_losses, axis=1)

    # tf_total_similarity_loss = 0
    # for i in range(config.MAX_ANSWER_WORDS - 1):
    #     tf_pair_similarity_loss = tf.reduce_sum(tf.multiply(tf_probabilities[:, i, :], tf_probabilities[:, i + 1, :]), axis=1)
    #     tf_total_similarity_loss += tf.reduce_mean(tf_pair_similarity_loss)

    tf_masked_losses = tf.multiply(tf_total_losses, tf_answer_masks)
    tf_total_loss = tf.reduce_mean(tf_masked_losses)
    #tf_total_loss += tf_total_similarity_loss  * SIMILARITY_LOSS_CONST
    #tf_total_loss = tf.reduce_mean(tf_total_losses) + tf_total_similarity_loss * SIMILARITY_LOSS_CONST

# Visualize
baseline_model_func.create_tensorboard_visualization('cic')

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf_total_loss)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
assert tf_embeddings not in tf.trainable_variables()
with tf.name_scope("SAVER"):
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)

model_io = {'questions': tf_question_indices, 'question_lengths': tf_question_lengths,
            'answers': tf_answer_indices, 'answer_masks': tf_answer_masks,
            'contexts': tf_context_indices, 'context_lengths': tf_context_lengths,
            'probabilities': tf_probabilities, 'predictions': tf_predictions,
            'batch_size': tf_batch_size}

# GRAPH EXECUTION ######################################################################################################

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
    sample_examples = examples[:sample_size]

    num_sample_context_empty_embs = 0
    num_sample_context_embs = 0
    assert len(sample_examples) == np_sample_context_embs.shape[0]
    assert config.MAX_CONTEXT_WORDS == np_sample_context_embs.shape[1]
    for example_index in range(np_sample_context_embs.shape[0]):
        num_context_tokens = len(sample_examples[example_index][2].split())
        for token_index in range(np_sample_context_embs.shape[1]):
            if token_index < num_context_tokens:
                num_sample_context_embs += 1
                if not np_sample_context_embs[example_index, token_index, :].any():
                    num_sample_context_empty_embs += 1
    frac_empty_sample_embeddings = num_sample_context_empty_embs / num_sample_context_embs

    # num_filled_embs = np.count_nonzero(np_sample_context_embs)
    # frac_filled_embs = 1 - (num_filled_embs / np_sample_context_embs.size)
    print('Fraction of words in context without embeddings: %s' % frac_empty_sample_embeddings)
    if USE_SPACY_NOT_GLOVE:
        for i in range(sample_size):
            context_tokens = contexts[i].split()
            for j in range(np_sample_context_embs.shape[1]):
                if j < len(context_tokens):
                    word = context_tokens[j]
                    word_vector = sdt.nlp(word).vector
                    word_index = vocab_dict[word]
                    stored_vector = np_embeddings[word_index, :]
                    if not np.isclose(word_vector, np.zeros([config.GLOVE_EMB_SIZE])).all():
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
                    if not np.isclose(word_vector, np.zeros([config.GLOVE_EMB_SIZE])).all():
                        assert np.isclose(np_sample_question_embs[i, j, :], word_vector).all()
                        assert np.isclose(np_sample_question_embs[i, j, :], stored_vector).all()
    print('Inputs validated')

# TRAINING #############################################################################################################

num_batches = int(num_examples * TRAIN_FRAC / BATCH_SIZE)
num_train_examples = BATCH_SIZE * num_batches
val_index_start = BATCH_SIZE * num_batches

print('Number of training examples: %s' % num_train_examples)

if RESTORE_FROM_SAVE:
    print('Restoring from save...')
    baseline_model_func.restore_model_from_save(config.BASELINE_MODEL_SAVE_DIR,
                                                var_list=tf.trainable_variables(),
                                                sess=sess)

np_train_predictions = None

if TRAIN_MODEL_BEFORE_PREDICTION:
    print('Training model...')
    all_train_predictions = []
    for epoch in range(NUM_EPOCHS):
        print('Epoch: %s' % epoch)
        losses = []
        sim_losses = []
        accuracies = []
        word_accuracies = []
        frac_zeros =[]
        for i in range(num_batches):
            np_question_batch = np_questions[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE, :]
            np_answer_batch = np_answers[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE, :]
            np_answer_mask_batch = np_answer_masks[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE, :]
            np_context_batch = np_contexts[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE, :]
            np_context_length_batch = np_context_lengths[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]
            np_question_length_batch = np_question_lengths[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]
            np_batch_predictions, np_loss, _, np_all_hidden_states = sess.run([tf_predictions, tf_total_loss, train_op, all_hidden_states],
                                                        feed_dict={tf_question_indices: np_question_batch,
                                                                   tf_question_lengths: np_question_length_batch,
                                                                   tf_answer_indices: np_answer_batch,
                                                                   tf_answer_masks: np_answer_mask_batch,
                                                                   tf_context_indices: np_context_batch,
                                                                   tf_context_lengths: np_context_length_batch,
                                                                   tf_batch_size: BATCH_SIZE})
            if epoch == NUM_EPOCHS - 1 and i == num_batches - 1:
                for j, hidden_state in enumerate(np_all_hidden_states):
                    print('Timestep: %s' % j)
                    print(hidden_state[0][0, :20])
                    print(hidden_state[1][0, :20])
            accuracy, word_accuracy = sdt.compute_mask_accuracy(np_answer_batch,
                                                                np_batch_predictions,
                                                                np_answer_mask_batch)
            frac_zero = sdt.compute_multi_label_accuracy(np_batch_predictions,
                                                         np.zeros([BATCH_SIZE, config.MAX_ANSWER_WORDS]))
            accuracies.append(accuracy)
            word_accuracies.append(word_accuracy)
            frac_zeros.append(frac_zero)
            if PRINT_ACCURACY_EVERY_N_BATCHES is not None and i % PRINT_ACCURACY_EVERY_N_BATCHES == 0:
                print('Batch TRAIN EM Score: %s' % np.mean(accuracies))
            losses.append(np_loss)
            if epoch == NUM_EPOCHS - 1:  # last epoch
                all_train_predictions.append(np_batch_predictions)
        epoch_loss = np.mean(losses)
        epoch_accuracy = np.mean(accuracies)
        epoch_word_accuracy = np.mean(word_accuracies)
        epoch_frac_zero = np.mean(frac_zeros)
        print('Epoch loss: %s' % epoch_loss)
        print('Epoch TRAIN EM Accuracy: %s' % epoch_accuracy)
        print('Epoch TRAIN Word Accuracy: %s' % epoch_word_accuracy)
        print('Epoch fraction of zero vector answers: %s' % epoch_frac_zero)
        print('Saving model %s' % epoch)
        saver.save(sess, config.BASELINE_MODEL_SAVE_DIR,
                   global_step=epoch)  # Save model after every epoch

    np_train_predictions = np.concatenate(all_train_predictions, axis=0)
    # total_train_accuracy = sdt.compute_multi_label_accuracy(np_train_predictions, np_answers[:val_index_start, :])
    # print('Final TRAIN EM Score: %s' % total_train_accuracy)
    print('Finished training')

if PREDICT_ON_TRAINING_EXAMPLES:
    print('Predicting on training examples...')
    np_train_predictions, np_train_probabilities = baseline_model_func.predict_on_examples(model_io,
                                                                   np_questions[:val_index_start, :],
                                                                   np_question_lengths[:val_index_start],
                                                                   np_contexts[:val_index_start, :],
                                                                   np_context_lengths[:val_index_start],
                                                                   BATCH_SIZE)

if PRINT_TRAINING_EXAMPLES and (TRAIN_MODEL_BEFORE_PREDICTION or PREDICT_ON_TRAINING_EXAMPLES):
    print('Printing training examples...')
    # Print training examples
    predictions = sdt.convert_numpy_array_answers_to_strings(np_train_predictions[:NUM_EXAMPLES_TO_PRINT, :],
                                                             contexts[:NUM_EXAMPLES_TO_PRINT],
                                                             answer_is_span=False, zero_stop_token=True)
    for i, each_prediction in enumerate(predictions):
        print('Prediction: %s' % each_prediction)
        print('Answer: %s' % examples[i][1])
        print('Answer array: %s' % np_answers[i, :])
        print('Prediction array: %s' % np_train_predictions[i, :])
        print('Question: %s' % examples[i][0])
        print('Len context: %s' % len(contexts[i]))
        print('Answer mask: %s' % np.around(np_answer_masks[i, :], decimals=1))

if PREDICT_ON_TRAINING_EXAMPLES or TRAIN_MODEL_BEFORE_PREDICTION:
    train_accuracy, train_word_accuracy = sdt.compute_mask_accuracy(np_answers[:val_index_start, :],
                                                        np_train_predictions,
                                                        np_answer_masks[:val_index_start, :])
    print('Total TRAIN EM Accuracy: %s' % train_accuracy)
    print('Total TRAIN Word Accuracy: %s' % train_word_accuracy)


# PREDICTION ###########################################################################################################

print('\n######################################\n')
print('Predicting...')

np_val_predictions, np_val_probabilities = baseline_model_func.predict_on_examples(model_io,
                                                             np_questions[val_index_start:, :],
                                                             np_question_lengths[val_index_start:],
                                                             np_contexts[val_index_start:, :],
                                                             np_context_lengths[val_index_start:],
                                                             BATCH_SIZE)

all_val_predictions = sdt.convert_numpy_array_answers_to_strings(np_val_predictions, contexts[val_index_start:],
                                                                 answer_is_span=False, zero_stop_token=True)

if SAVE_VALIDATION_PREDICTIONS:
    print('Saving validation predictions to: %s' % config.VALIDATION_PREDICTIONS_FILE)

    pkl.dump([examples[val_index_start:], all_val_predictions], open(config.VALIDATION_PREDICTIONS_FILE, 'wb'))

if PRODUCE_OUTPUT_PREDICTIONS_FILE:
    print('Saving submission predictions file to: %s' % config.SUBMISSION_PREDICTIONS_FILE)
    final_predictions = {}
    for i in range(num_examples):
        current_prediction = all_val_predictions[i]
        current_id = examples[i][3]
        current_answer = answers[i]
        final_predictions[current_id] = sdt.detokenize_string(current_prediction)
    with open(config.SUBMISSION_PREDICTIONS_FILE, 'w') as json_file:
        json.dump(final_predictions, json_file)

if PRINT_VALIDATION_EXAMPLES:
    predictions \
        = sdt.convert_numpy_array_answers_to_strings(np_val_predictions[:NUM_EXAMPLES_TO_PRINT, :],
                                                     contexts[val_index_start:val_index_start + NUM_EXAMPLES_TO_PRINT],
                                                     answer_is_span=False, zero_stop_token=True)
    # Print validation examples
    for i, each_prediction in enumerate(predictions):
        print('Prediction: %s' % each_prediction)
        print('Answer: %s' % examples[val_index_start + i][1])
        print('Answer array: %s' % np_answers[val_index_start + i, :])
        print('Prediction array: %s' % np_val_predictions[i, :])
        print('Question: %s' % examples[val_index_start + i][0])
        print('Len context: %s' % len(contexts[val_index_start + i]))
        print('Context: %s' % contexts[val_index_start + i])
        print()

if not SUBMISSION_MODE:
    val_accuracy, val_word_accuracy = sdt.compute_mask_accuracy(np_val_predictions,
                                                                np_answers[val_index_start:],
                                                                np_answer_masks[val_index_start:])

    print('VAL EM Accuracy: %s' % val_accuracy)
    print('VAL Word Accuracy: %s' % val_word_accuracy)
    print('Prediction shape: %s' % str(np_val_predictions.shape))



















