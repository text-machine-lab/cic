"""Trains and predicts using the baseline LSTM model."""
print('Starting program')
import tensorflow as tf
import squad_dataset_tools as sdt
import config

NUM_PARAGRAPHS = 1000  # 18896
print('Loading SQuAD dataset')
paragraphs = sdt.load_squad_dataset_from_file(config.SQUAD_TRAIN_SET)
sample_paragraphs = paragraphs[:NUM_PARAGRAPHS]
print('Processing %s paragraphs...' % len(sample_paragraphs))
print('Tokenizing paragraph samples')
tk_paragraphs = sdt.tokenize_paragraphs(sample_paragraphs)
print('Building vocabulary')
vocabulary = sdt.generate_vocabulary_for_paragraphs(tk_paragraphs)
print('Flatting paragraphs into examples')
examples = sdt.convert_paragraphs_to_flat_format(tk_paragraphs)
print('Converting each example to numpy arrays')
np_questions, np_answers, np_contexts, ids, np_as \
    = sdt.generate_numpy_features_from_squad_examples(examples, vocabulary.token2id)
print('Number of examples: %s' % np_questions.shape[0])






