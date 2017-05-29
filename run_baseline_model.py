"""Trains and predicts using the baseline LSTM model."""
print('Starting program')
import tensorflow as tf
import squad_dataset_tools as sdt
import config
import baseline_model
import pprint

LEARNING_RATE = .00001
NUM_PARAGRAPHS = 10 # 18896
LSTM_HIDDEN_DIM = 400

# Don't forget to tokenize!

print('Loading SQuAD dataset')
paragraphs = sdt.load_squad_dataset_from_file(config.SQUAD_TRAIN_SET)
sample_paragraphs = paragraphs[:NUM_PARAGRAPHS]
print('Processing %s paragraphs...' % len(sample_paragraphs))
print('Tokenizing paragraph samples')
tk_paragraphs = sample_paragraphs # sdt.tokenize_paragraphs(sample_paragraphs)
print('Building vocabulary')
vocabulary = sdt.generate_vocabulary_for_paragraphs(tk_paragraphs).token2id
vocabulary_size = len(vocabulary)
print('Len vocabulary: %s' % vocabulary_size)
print('Flatting paragraphs into examples')
examples = sdt.convert_paragraphs_to_flat_format(tk_paragraphs)
print('Converting each example to numpy arrays')
np_questions, np_answers, np_contexts, ids, np_as \
    = sdt.generate_numpy_features_from_squad_examples(examples, vocabulary)
num_examples = np_questions.shape[0]
print('Number of examples: %s' % np_questions.shape[0])
print('Loading embeddings for each word in vocabulary')
np_embeddings = sdt.construct_embeddings_for_vocab(vocabulary)
print('Loading embeddings into Tensorflow')
tf_embeddings = tf.Variable(np_embeddings, name='word_embeddings', dtype=tf.float32)
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

tf_vocab_w = tf.Variable(tf.random_normal([LSTM_HIDDEN_DIM, vocabulary_size]))
tf_vocab_b = tf.Variable(tf.random_normal([vocabulary_size]))

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
    tf_output_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=each_prediction, labels=tf_answer_indices[:, i])
    tf_total_loss += tf_output_loss

sess = tf.InteractiveSession()
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf_total_loss)
init = tf.global_variables_initializer()
sess.run(init)

np_predictions_indices = sess.run(tf_predictions_indices, feed_dict={tf_question_indices: np_questions,
                                                                     tf_answer_indices: np_answers,
                                                                     tf_context_indices: np_contexts,
                                                                     tf_batch_size: num_examples})

print(np_predictions_indices.shape)
pprint.pprint(np_predictions_indices)


















