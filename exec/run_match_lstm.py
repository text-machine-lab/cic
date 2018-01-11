"""Trains and predicts a model using a Match-LSTM and Answer Pointer. Design based on this paper:
https://arxiv.org/pdf/1608.07905.pdf. Model is not currently working, should implement teacher-forcing!"""
import json
import pickle as pkl
import sys

import numpy as np
import os
import random
from cic.utils import squad_tools as sdt
from cic.models import match_lstm
from cic.models.match_lstm import LEARNING_RATE, NUM_PARAGRAPHS, RNN_HIDDEN_DIM, NUM_EXAMPLES_TO_PRINT, \
    TRAIN_FRAC, \
    VALIDATE_PROPER_INPUTS, RESTORE_FROM_SAVE, TRAIN_MODEL_BEFORE_PREDICTION, PREDICT_ON_TRAINING_EXAMPLES, NUM_EPOCHS, \
    PRINT_TRAINING_EXAMPLES, PRINT_VALIDATION_EXAMPLES, PRINT_ACCURACY_EVERY_N_BATCHES, BATCH_SIZE, KEEP_PROB, \
    STOP_TOKEN_REWARD, TURN_OFF_TF_LOGGING, USE_SPACY_NOT_GLOVE, SHUFFLE_EXAMPLES, SAVE_VALIDATION_PREDICTIONS, \
    PRODUCE_OUTPUT_PREDICTIONS_FILE, REMOVE_EXAMPLES_GREATER_THAN_MAX_LENGTH, \
    REMOVE_EXAMPLES_WITH_MIN_FRAC_EMPTY_EMBEDDINGS, SEED

from cic import config

sdt.initialize_nlp()

# CONTROL PANEL ########################################################################################################

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

if SEED is not None:
    random.seed(SEED)

if not USE_SPACY_NOT_GLOVE:
    config.GLOVE_EMB_SIZE = 200

if not os.path.exists(config.BASELINE_MODEL_SAVE_DIR):
    os.makedirs(config.BASELINE_MODEL_SAVE_DIR)

if TURN_OFF_TF_LOGGING:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

frac_answer_lengths_one = np.mean(np_answer_lengths == 1)
print('Fraction of answers of length one: %s' % frac_answer_lengths_one)

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

qa_model = match_lstm.LSTMBaselineModel(RNN_HIDDEN_DIM, LEARNING_RATE,
                                        save_dir=config.BASELINE_MODEL_SAVE_DIR,
                                        restore_from_save=RESTORE_FROM_SAVE)

# Visualize
match_lstm.create_tensorboard_visualization('cic')

# GRAPH EXECUTION ######################################################################################################

if VALIDATE_PROPER_INPUTS:
    print('Validating proper inputs')
    # Validates that embedding lookups for the first 1000 contexts and questions
    # look up the correct embedding for each index, and that the embedding is
    # the same as that stored in the embedding table and in the spacy nlp object.

    sample_size = 1000
    if sample_size > num_examples:
        sample_size = num_examples
    fd={qa_model.tf_context_indices: np_contexts[:sample_size, :], qa_model.tf_embeddings: np_embeddings}
    np_sample_context_embs = qa_model.tf_context_embs.eval(fd)
    sample_examples = examples[:sample_size]
    print('I am here.')
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
        fd={qa_model.tf_question_indices: np_questions[:sample_size, :], qa_model.tf_embeddings: np_embeddings}
        np_sample_question_embs = qa_model.tf_question_embs.eval(fd)
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

if TRAIN_MODEL_BEFORE_PREDICTION:
    np_train_predictions = qa_model.train(np_embeddings,
                                          np_questions[:val_index_start, :],
                                          np_contexts[:val_index_start, :],
                                          np_answers[:val_index_start, :],
                                          np_answer_masks[:val_index_start, :],
                                          BATCH_SIZE, NUM_EPOCHS, KEEP_PROB,
                                          print_per_n_batches=PRINT_ACCURACY_EVERY_N_BATCHES)

if PREDICT_ON_TRAINING_EXAMPLES:
    print('Predicting on training examples...')
    np_train_predictions, np_train_probabilities = qa_model.predict_on_examples(np_embeddings,
                                                                                np_questions[:val_index_start, :],
                                                                                np_contexts[:val_index_start, :],
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
    print(np_answers[:val_index_start, :].shape)
    print(np_train_predictions.shape)
    train_accuracy, train_word_accuracy = sdt.compute_mask_accuracy(np_answers[:val_index_start, :],
                                                        np_train_predictions,
                                                        np_answer_masks[:val_index_start, :])
    print('Total TRAIN EM Accuracy: %s' % train_accuracy)
    print('Total TRAIN Word Accuracy: %s' % train_word_accuracy)


# PREDICTION ###########################################################################################################

print('\n######################################\n')
print('Predicting...')

np_val_predictions, np_val_probabilities = qa_model.predict_on_examples(np_embeddings,
                                                                        np_questions[val_index_start:, :],
                                                                        np_contexts[val_index_start:, :],
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



















