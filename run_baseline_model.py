"""Trains and predicts using the baseline LSTM model."""
print('Starting program')
import tensorflow as tf
import squad_dataset_tools as sdt
import config
import baseline_model
import pprint
import numpy as np

LEARNING_RATE = .00005
NUM_PARAGRAPHS = 50
LSTM_HIDDEN_DIM = 1600
NUM_EPOCHS = 100
NUM_EXAMPLES_TO_PRINT = 10

print('Loading SQuAD dataset')
paragraphs = sdt.load_squad_dataset_from_file(config.SQUAD_TRAIN_SET)
training_paragraphs = paragraphs[:NUM_PARAGRAPHS]
print('Processing %s paragraphs...' % len(training_paragraphs))
print('Tokenizing paragraph samples')
tk_paragraphs = sdt.tokenize_paragraphs(training_paragraphs)
print('Building vocabulary')
vocab_dict = sdt.generate_vocabulary_for_paragraphs(tk_paragraphs).token2id
vocabulary = sdt.invert_dictionary(vocab_dict)  # becomes list of words essentially
vocabulary_size = len(vocab_dict)
print('Len vocabulary: %s' % vocabulary_size)
print('Flatting paragraphs into examples')
examples = sdt.convert_paragraphs_to_flat_format(tk_paragraphs)
print('Converting each example to numpy arrays')
np_questions, np_answers, np_contexts, ids, np_as \
    = sdt.generate_numpy_features_from_squad_examples(examples, vocab_dict, answer_indices_from_context=True)
print('Maximum index in answers should be less than max context size + 1: %s' % np_answers.max())
num_examples = np_questions.shape[0]
print('Number of examples: %s' % np_questions.shape[0])
print('Loading embeddings for each word in vocabulary')
np_embeddings = sdt.construct_embeddings_for_vocab(vocab_dict)

num_embs = np_embeddings.shape[0]
emb_size = np_embeddings.shape[1]
num_empty_embs = 0
print(np_embeddings[0, :].shape)
for i in range(num_embs):
    if np.isclose(np_embeddings[i, :], np.zeros([emb_size])).all():
        num_empty_embs += 1
fraction_empty_embs = num_empty_embs / num_embs
print('Fraction of empty embeddings in vocabulary: %s' % fraction_empty_embs)

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
tf_context_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_CONTEXT_WORDS), name='context_indices')
tf_answer_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_ANSWER_WORDS), name='answer_indices')
tf_batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')

tf_question_embs = tf.nn.embedding_lookup(tf_embeddings, tf_question_indices, name='question_embeddings')
tf_context_embs = tf.nn.embedding_lookup(tf_embeddings, tf_context_indices, name='context_embeddings')
print(tf_question_embs.shape)
print(tf_context_embs.shape)

# Model
_, question_hidden_states = baseline_model.build_gru(LSTM_HIDDEN_DIM, tf_batch_size,
                                                     [tf_question_embs], config.MAX_QUESTION_WORDS,
                                                     gru_scope='QUESTION_RNN')
_, context_hidden_states = baseline_model.build_gru(LSTM_HIDDEN_DIM, tf_batch_size,
                                                    [tf_context_embs], config.MAX_CONTEXT_WORDS,
                                                    gru_scope='CONTEXT_RNN')

question_context_emb = tf.concat([question_hidden_states[-1], context_hidden_states[-1]], axis=1)

answer_outputs, _ = baseline_model.build_gru(LSTM_HIDDEN_DIM, tf_batch_size,
                                             [], config.MAX_ANSWER_WORDS, time_step_inputs=[question_context_emb],
                                             gru_scope='ANSWER_RNN')

tf_vocab_w = tf.Variable(tf.random_normal([LSTM_HIDDEN_DIM, config.MAX_CONTEXT_WORDS + 1]))
tf_vocab_b = tf.Variable(tf.random_normal([config.MAX_CONTEXT_WORDS + 1]))

vocab_predictions = []
for each_output in answer_outputs:
    word_prediction = tf.matmul(each_output, tf_vocab_w) + tf_vocab_b
    vocab_predictions.append(word_prediction)

predictions_indices = []
for each_prediction in vocab_predictions:
    tf_prediction_indices = tf.argmax(each_prediction, axis=1)
    predictions_indices.append(tf.reshape(tf_prediction_indices, [-1, 1]))

tf_predictions_indices = tf.concat(predictions_indices, axis=1)

print('Model output prediction indices shape: %s' % str(tf_predictions_indices.get_shape()))
print('RNN output shape: %s' % str(answer_outputs[0].get_shape()))
print('Labels indices for loss function: %s' % str(tf_answer_indices[:, 0].get_shape()))
print('Prediction distribution shape: %s' % str(vocab_predictions[0].get_shape()))
# Loss
tf_total_loss = 0
for i, each_prediction in enumerate(vocab_predictions):
    tf_output_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=each_prediction, labels=tf_answer_indices[:, i])
    tf_output_loss = tf.reduce_mean(tf_output_losses, axis=0)
    if i == 0:
        print('Shape of output losses for single word: %s' % str(tf_output_losses.get_shape()))
    tf_total_loss += tf_output_loss

sess = tf.InteractiveSession()
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf_total_loss)
init = tf.global_variables_initializer()
sess.run(init)

print(np_embeddings)

print('Training model...')
batch_size = 50
num_batches = num_examples // batch_size
for epoch in range(NUM_EPOCHS):
    print('Epoch: %s' % epoch)
    losses = []
    for i in range(num_batches):
        np_question_batch = np_questions[i*batch_size:i*batch_size+batch_size, :]
        np_answer_batch = np_answers[i*batch_size:i*batch_size+batch_size, :]
        np_context_batch = np_contexts[i*batch_size:i*batch_size+batch_size, :]
        np_predictions_batch, np_loss, _ = sess.run([tf_predictions_indices, tf_total_loss, train_op],
                                                    feed_dict={tf_question_indices: np_question_batch,
                                                               tf_answer_indices: np_answer_batch,
                                                               tf_context_indices: np_context_batch,
                                                               tf_batch_size: batch_size})
        losses.append(np_loss)
    epoch_loss = np.mean(losses)
    print('Epoch loss: %s' % epoch_loss)

print('Predicting...')
np_predictions_indices = sess.run(tf_predictions_indices, feed_dict={tf_question_indices: np_questions,
                                                                     tf_answer_indices: np_answers,
                                                                     tf_context_indices: np_contexts,
                                                                     tf_batch_size: num_examples})

contexts = [example[2] for example in examples]

predictions = sdt.convert_numpy_array_answers_to_strings(np_predictions_indices[:NUM_EXAMPLES_TO_PRINT, :], contexts[:NUM_EXAMPLES_TO_PRINT])
for i, each_prediction in enumerate(predictions):
    print('Prediction: %s' % each_prediction)
    print('Answer: %s' % examples[i][1])
    print('Prediction array: %s' % np_predictions_indices[i, :])
    print('Answer array: %s' % np_answers[i, :])
    print('Question: %s' % examples[i][0])

print()

print(np_predictions_indices.shape)
# pprint.pprint(np_predictions_indices)


















