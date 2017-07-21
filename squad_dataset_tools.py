"""Copyright 2017 David Donahue. LSTM baseline for SQuAD dataset. Reads question with LSTM.
Reads passage with LSTM. Outputs answer with LSTM."""
import config
import json
import spacy
import numpy as np
import string
import re
import gensim
import unittest2

nlp = None


def initialize_nlp():
    global nlp
    nlp = spacy.load('en_vectors_glove_md')  # python -m spacy download en

def load_squad_dataset_from_file(squad_filename):
    all_paragraphs = []
    all_qas = []
    with open(squad_filename) as squad_file:
        all = json.load(squad_file)
        data = all['data']
        for each_document in data:
            for each_paragraph in each_document['paragraphs']:
                all_paragraphs.append(each_paragraph)
    return all_paragraphs


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def detokenize_string(s):
    s_tokens = s.split()
    s_tokens_final = []
    for i in range(len(s_tokens)):
        if s_tokens[i] != ["'"] and "'" in s_tokens[i] and i > 0:
            s_tokens_final[-1] += s_tokens[i]
        else:
            s_tokens_final.append(s_tokens[i])
    return ' '.join(s_tokens_final)


def convert_paragraphs_to_flat_format(paragraphs):
    """Converts a series of paragraphs from SQuAD dataset,
    into a list of (question, answer, context, id, answer_start) tuples. Returns an entry per question, per answer to each paragraph.

    paragraphs - a list of {'context', 'qas'} dictionaries where context is the paragraph and qas is a list of
    {'answers', 'question', 'id'} tuples

    Returns: a list of (question, answer, context, id, answer_start) tuples"""
    qacs_tuples = []
    for each_paragraph in paragraphs:
        context = each_paragraph['context']
        for each_qas in each_paragraph['qas']:
            question = each_qas['question']
            id = each_qas['id']
            for each_answer in each_qas['answers']:
                answer = each_answer['text']
                answer_start = each_answer['answer_start']
                # Add tuple
                qacs_tuples.append((question, answer, context, id, answer_start))
    return qacs_tuples


def remove_excess_spaces_from_paragraphs(paragraphs):
    """Uses split and join to remove all excess spaces from paragraphs."""
    clean_paragraphs = []
    for each_paragraph in paragraphs:
        clean_paragraph = {}
        clean_paragraph['context'] = ' '.join(each_paragraph['context'].split())
        clean_paragraph['qas'] = []
        for each_qas in each_paragraph['qas']:
            clean_qas = {}
            clean_qas['question'] = ' '.join(each_qas['question'].split())
            clean_qas['id'] = each_qas['id']
            clean_qas['answers'] = []
            for each_answer in each_qas['answers']:
                clean_answer = {}
                clean_answer['text'] = ' '.join(each_answer['text'].split())
                clean_answer['answer_start'] = each_answer['answer_start']

                clean_qas['answers'].append(clean_answer)
            clean_paragraph['qas'].append(clean_qas)
        clean_paragraphs.append(clean_paragraph)
    return clean_paragraphs


def convert_numpy_array_to_strings(np_examples, vocabulary, stop_token=None, keep_stop_token=False):
    """Converts a numpy array of indices into a list of strings.

    np_examples - m x n numpy array of ints, where m is the number of
    strings encoded by indices, and n is the max length of each string
    vocab_dict - where vocab_dict[index] gives a word in the vocabulary
    with that index

    Returns: a list of strings, where each string is constructed from
    indices in the array as they appear in the vocabulary."""
    assert stop_token is not None or not keep_stop_token
    m = np_examples.shape[0]
    n = np_examples.shape[1]
    examples = []
    for i in range(m):
        each_example = ''
        for j in range(n):
            word_index = np_examples[i][j]
            word = vocabulary[word_index]
            if stop_token is not None and word == stop_token:
                if keep_stop_token:
                    if j > 0:
                        each_example += ' '
                    each_example += stop_token
                break
            if j > 0 and word != '':
                each_example += ' '
            each_example += word
        examples.append(each_example)
    return examples


def convert_numpy_array_answers_to_strings(np_answers, contexts, answer_is_span=False, zero_stop_token=False):
    """Converts a numpy array of answer indices into a list of strings.
    Indices are taken from the context of each answer.

    np_examples - m x n numpy array of ints, where m is the number of
    strings encoded by indices, and n is the max length of each string
    vocab_dict - where vocab_dict[index] gives a word in the vocabulary
    with that index
    contexts - list of contexts, one for each answer
    answer_is_span - answer array contains start and end indices of array,
    instead of sequence of indices
    zero_stop_token - function stops reconstruction of string upon
    reading stop token (index 0)

    Returns: a list of strings, where each string is constructed from
    indices in the array as they appear in the context."""
    m = np_answers.shape[0]
    n = np_answers.shape[1]
    assert m == len(contexts)
    answer_strings = []
    for i in range(m):
        context_tokens = contexts[i].split()
        if answer_is_span:
            start_index = np_answers[i, 0]
            end_index = np_answers[i, 1]
            answer_string = ' '.join(context_tokens[start_index:end_index+1])
        else:
            answer_string = ''
            for j in range(n):
                if np_answers[i, j] != 0:
                    context_index = np_answers[i, j] - 1  # was shifted for index 0 -> ''
                    if context_index < len(context_tokens):
                        if j > 0:
                            answer_string += ' '
                        answer_string += context_tokens[context_index]
                else:
                    if zero_stop_token:
                        break
        answer_strings.append(answer_string)
    return answer_strings


def invert_dictionary(dictionary):
    inv_dictionary = {v: k for k, v in dictionary.items()}
    return inv_dictionary


def look_up_glove_embeddings(vocab_dict, glove_emb_path=config.GLOVE_200_FILE):
    """Find a GloVe embedding for each word in
    index_to_word, if it exists. Create a dictionary
    mapping from words to GloVe vectors and return it."""
    word_to_glove = {}
    with open(glove_emb_path, 'rb') as f:
        for line in f:
            line_tokens = line.split()
            glove_word = line_tokens[0].lower().decode('utf-8')
            if glove_word in vocab_dict.keys():
                glove_emb = [float(line_token) for line_token in line_tokens[1:]]
                word_to_glove[glove_word] = glove_emb

    return word_to_glove


def construct_embeddings_for_vocab(vocab_dict, use_spacy_not_glove=True):
    """Creates matrix to hold embeddings for words in vocab, ordered by index in vocabulary.
    Intended to be used as lookup embedding tensor.

    Returns: embedding matrix for all words in the vocab with an associated embedding."""
    word_to_glove = {}
    if not use_spacy_not_glove:
        word_to_glove = look_up_glove_embeddings(vocab_dict)
        word_to_glove[''] = np.zeros([config.GLOVE_EMB_SIZE])
        print('Word_to_glove size: %s' % len(word_to_glove))
    m = len(vocab_dict.keys())
    np_embeddings = np.zeros([m, config.GLOVE_EMB_SIZE])
    for each_word in vocab_dict:
        index = vocab_dict[each_word]
        if use_spacy_not_glove:
            embedding = nlp(each_word).vector
        else:
            if each_word in word_to_glove.keys():
                embedding = word_to_glove[each_word]
            else:
                embedding = np.zeros([config.GLOVE_EMB_SIZE])
        #print(embedding.vector)
        np_embeddings[index, :] = embedding

    return np_embeddings


def tokenize_paragraphs(paragraphs):
    """Tokenizes paragraphs using spacy module and returns a copy of them.

    nlp - Spacy object able to tokenize a string using nlp.tokenizer(string)
    """
    tk_paragraphs = []
    for each_paragraph in paragraphs:
        tk_paragraph = {}
        context = each_paragraph['context']
        tk_context_doc = nlp.tokenizer(context)
        tk_context = ' '.join([str(word) for word in tk_context_doc]).lower()

        tk_paragraph['context'] = tk_context
        tk_paragraph['qas'] = []
        for each_qas in each_paragraph['qas']:
            question = each_qas['question']
            tk_question_doc = nlp.tokenizer(question)
            tk_question = ' '.join([str(word) for word in tk_question_doc]).lower()

            id = each_qas['id']
            tk_answers = []
            for each_answer in each_qas['answers']:
                answer = each_answer['text']
                tk_answer_doc = nlp.tokenizer(answer)
                tk_answer = ' '.join([str(word) for word in tk_answer_doc]).lower()
                answer_start = each_answer['answer_start']
                tk_answers.append({'text': tk_answer, 'answer_start': answer_start})
            tk_paragraph['qas'].append({'question': tk_question, 'id': id, 'answers': tk_answers})
        tk_paragraphs.append(tk_paragraph)
    return tk_paragraphs


class holder:
    def __init__(self, arg_vocab_dict):
        self.token2id = arg_vocab_dict


def generate_vocabulary_for_paragraphs(paragraphs):
    """Generates a token-based vocabulary using gensim for all words in contexts, questions and answers of
    'paragraphs'. Converts all tokens to lowercase

    paragraphs - a list of {'context', 'qas'} dictionaries where context is the paragraph and qas is a list of
    {'answers', 'question', 'id'} tuples
    prune_at - maximum size of vocabulary, infrequent words will be pruned after this level

    Returns: gensim.corpora.Dictionary object containing vocabulary. Can view dictionary mapping with self.token2id."""
    documents = []
    for each_paragraph in paragraphs:
        context = each_paragraph['context']
        documents.append(context.lower().split())
        for each_qas in each_paragraph['qas']:
            question = each_qas['question']
            documents.append(question.lower().split())
            # for each_answer in each_qas['answers']:
            #     answer = each_answer['text']
            #     documents.append(answer.lower().split())
    vocabulary = ['']
    vocab_dict = {}
    for document in documents:
        doc_tokens = document
        for doc_token in doc_tokens:
            if doc_token not in vocabulary:
                vocabulary.append(doc_token)
    for i in range(len(vocabulary)):
        vocab_dict[vocabulary[i]] = i

    return holder(vocab_dict)


def generate_numpy_features_from_squad_examples(examples, vocab_dict,
                                                max_question_words=config.MAX_QUESTION_WORDS,
                                                max_answer_words=config.MAX_ANSWER_WORDS,
                                                max_context_words=config.MAX_CONTEXT_WORDS,
                                                answer_indices_from_context=False,
                                                answer_is_span=False):
    """Uses a list of squad QA examples to generate features for a QA model. Features are numpy arrays for
    questions, answers, and contexts, where each word is represented as an index in a vocabulary. Answer
    indices can either be taken from general vocabulary or context vocabulary.

    examples - list of (question, answer, context, id, answer_start) tuples
    vocab_dict - mapping from words to indices in vocabulary
    max_question_words - max length of vector containing question word ids
    max_answer_words - max length of vector containing answer word ids
    max_context_words - max length of vector containing context word ids
    answer_indices_from_context - if true, words in each answer will be represented as the indices of matching words
    in the context (+1 for each index, leaving room for index 0 -> ''). Answers should appear exactly in context
    Otherwise, words in each answer will be represented as indices from vocabulary vocab_dict
    answer_is_span - if true, each answer will be two indices, one for the index of the first token in the answer,
    and one for the index of the second token. Can only be true if answer_indices_from_context is true

    Returns: where m is the number of examples, an m x max_question_words array for questions, an m x max_answer_words
    array for answers, and an m x max_context_words array for context, an m-dimensional vector
    for question ids and an m by 2 vector for answer_starts/ends. Ie. returns (np_questions, np_answers, np_contexts, np_ids, np_as)."""
    if answer_is_span:
        assert answer_indices_from_context
    m = len(examples)
    np_questions = np.zeros([m, max_question_words], dtype=int)
    if answer_is_span:
        np_answers = np.zeros([m, 2], dtype=int)
    else:
        np_answers = np.zeros([m, max_answer_words], dtype=int)
    np_contexts = np.zeros([m, max_context_words], dtype=int)
    ids = []
    np_as = np.zeros([m, 2])

    for i, each_example in enumerate(examples):
        question, answer, context, id, answer_start = each_example
        question_tokens = question.lower().split()
        answer_tokens = answer.lower().split()
        context_tokens = context.lower().split()
        answer_end = answer_start + len(answer_tokens)
        for j, each_token in enumerate(question_tokens):
            if j < max_question_words and each_token in vocab_dict:
                np_questions[i, j] = vocab_dict[each_token]
        for j, each_token in enumerate(context_tokens):
            if j < max_context_words and each_token in vocab_dict:
                np_contexts[i, j] = vocab_dict[each_token]

        if answer_indices_from_context:
            for context_index, each_context_token in enumerate(context_tokens):
                answer_starts_here = True
                for answer_index, each_answer_token in enumerate(answer_tokens):
                    if context_index + answer_index >= len(context_tokens) or context_tokens[context_index + answer_index] != each_answer_token:
                        answer_starts_here = False
                if answer_starts_here:
                    if answer_is_span:
                        if context_index + len(answer_tokens) - 1 < config.MAX_CONTEXT_WORDS:
                            np_answers[i, 0] = context_index
                            np_answers[i, 1] = context_index + len(answer_tokens) - 1
                    else:
                        for answer_index in range(len(answer_tokens)):
                            if answer_index < config.MAX_ANSWER_WORDS and context_index + answer_index < config.MAX_CONTEXT_WORDS:
                                np_answers[i, answer_index] = context_index + answer_index + 1  # index 0 -> ''
                    break
        else:
            for j, each_token in enumerate(answer_tokens):
                if j < max_answer_words and each_token in vocab_dict:
                    np_answers[i, j] = vocab_dict[each_token] # index 0 -> ''
        ids.append(id)
        np_as[i, 0] = answer_start
        np_as[i, 1] = answer_end

    return np_questions, np_answers, np_contexts, ids, np_as


def compute_multi_label_accuracy(np_first, np_second):
    """Assume that inputs are span predictions and labels, and compute similarity score.

    np_first, np_second - m x 2 matrices, where m is the number examples, each example
    associated with two indices indicating the start and end of the span

    Returns: a float accuracy between 0 and 1, 1 indicating the matrices are the same."""
    assert np_first.shape == np_second.shape
    m = np_first.shape[0]
    num_correct = 0

    for index in range(m):
        if np.array_equal(np_first[index, :], np_second[index, :]):
            num_correct += 1
    accuracy = num_correct / m

    return accuracy


def compute_mask_accuracy(np_first, np_second, np_mask):
    """Computes accuracy between two numpy arrays, given a mask
    array that decides which answers to include in accuracy score.
    Accuracy score is exact match per row, and also returns
    per element accuracy with mask.

    np_first, np_second - arrays to compare
    np_mask - same size as np_first, np_second, with fractional weights
    per element indicating if it should be included in accuracy prediction

    Returns: Accuracies."""
    assert np_first.shape == np_second.shape
    m = np_first.shape[0]
    num_correct = 0

    rows_correct = 0
    np_same = (np_first == np_second)
    np_same_important = np.multiply(np_same, np_mask)
    element_wise_accuracy = np.sum(np_same_important) / np.sum(np_mask)
    for row in range(m):
        mask_sum = np.sum(np_mask[row, :])
        row_sum = np.sum(np_same_important[row, :])
        if mask_sum != 0:
            if row_sum / mask_sum > .9999:
                rows_correct += 1
    accuracy = rows_correct / m
    return accuracy, element_wise_accuracy


def compute_answer_mask(np_answers, stop_token=True, zero_weight=(1 / (config.MAX_CONTEXT_WORDS + 1))):
    """Returns a numpy 2D array filled with ones up to
    and including the first zero in each row of np_answers. This zero
    is interpretted as the stop token and all remaining
    tokens are not included in the mask. These tokens are set to
    zero in the output array. If all tokens are zeros in input row,
    returns all ones in that row.

    Returns: mask across relevant tokens in input array."""
    m = np_answers.shape[0]
    n = np_answers.shape[1]
    np_non_zeros = np.not_equal(np_answers, np.zeros([m, n]))
    print(np_non_zeros)
    np_mask = np.ones([m, n])
    for i in range(m):
        np_zeros = np.zeros([n])
        if np.array_equal(np_non_zeros[i, :], np_zeros):
            np_mask[i, :] = np_zeros  # no reward for empty answer
        else:
            for j in range(n-1):
                if not np_non_zeros[i, j]:
                    if stop_token:
                        np_mask[i, j] = zero_weight
                        np_mask[i, (j+1):] = 0
                    else:
                        np_mask[i, j:] = zero_weight
                    break
    return np_mask


class LSTM_Baseline_Test(unittest2.TestCase):
    def setUp(self):
        self.answer = {'text': 'once upon', 'answer_start': 69}
        self.answer2 = {'text': 'a time there was', 'answer_start': 123}
        self.answer3 = {'text': 'a useless paragraph.', 'answer_start': 0}
        self.qas1 = {'question': 'can birds fly?', 'id': 54321, 'answers': [self.answer, self.answer2]}
        self.qas2 = {'question': 'what is the meaning of life?', 'id': 5454, 'answers': [self.answer3]}
        self.paragraph = {'context': 'once upon a time there was a useless paragraph. The end.', 'qas': [self.qas1, self.qas2]}

    def test_compute_answer_mask(self):
        np_answers = np.array([[1, 2, 0, 0], [5, 0, 0, 0], [6, 7, 8, 0], [0, 0, 0, 0], [1, 2, 3, 4]])
        np_mask = compute_answer_mask(np_answers)
        print(np_mask)
        np_zeros = np.zeros([5, 6])
        assert np.array_equal(compute_answer_mask(np_zeros), np_zeros)
        # assert np.array_equal(np_mask, np.array([[1, 1, .005, 0], [1, .005, 0, 0], [1, 1, 1, 1], [.005, 0, 0, 0], [1, 1, 1, 1]]))

    def test_load_squad_dataset_from_file(self):
        all_paragraphs = load_squad_dataset_from_file(config.SQUAD_TRAIN_SET)
        assert all_paragraphs is not None
        assert len(all_paragraphs) > 0
        assert isinstance(all_paragraphs[0]['context'], str)
        assert isinstance(all_paragraphs[0]['qas'], list)

    def test_convert_paragraphs_to_flat_format(self):

        qacs_tuples = convert_paragraphs_to_flat_format([self.paragraph])
        assert ('can birds fly?', 'once upon', 'once upon a time there was a useless paragraph. The end.', 54321, 69) in qacs_tuples
        assert ('what is the meaning of life?', 'a useless paragraph.', 'once upon a time there was a useless paragraph. The end.', 5454, 0) in qacs_tuples
        #pprint.pprint(qacs_tuples)

    def test_generate_vocabulary_for_paragraphs(self):
        vocab = generate_vocabulary_for_paragraphs([self.paragraph])
        vocab_dict = vocab.token2id
        sample = (self.paragraph['context'] + ' ' + self.answer['text'] + ' ' + self.qas1['question']).lower().split()
        for word in sample:
            assert word in vocab_dict
        #print(vocab_dict)

    def test_tokenize_paragraphs(self):
        [tk_paragraph] = tokenize_paragraphs([self.paragraph])
        #print('Compare untokenized and tokenized paragraphs:')
        #pprint.pprint(self.paragraph)
        #pprint.pprint(tk_paragraph)

    def test_generate_numpy_features_from_squad_examples_empty(self):
        """Test empty list of examples."""
        np_questions, np_answers, np_contexts, ids, np_as \
            = generate_numpy_features_from_squad_examples([], {})
        assert np_questions.shape == (0, config.MAX_QUESTION_WORDS)
        assert np_answers.shape == (0, config.MAX_ANSWER_WORDS)
        assert np_contexts.shape == (0, config.MAX_CONTEXT_WORDS)
        #print(np_ids.shape)
        assert len(ids) == 0
        assert np_as.shape == (0, 2)

    def test_generate_numpy_features_from_squad_examples_single(self):
        """Generate single example and test the output."""
        question = 'what is the meaning of life ?'
        answer = 'life is a confusing place'
        context = 'life is an in deterministic sloppy joe'
        question_tokens = question.split()
        answer_tokens = answer.split()
        context_tokens = context.split()
        example = (question, answer, context, 1234, 5678)
        vocab_dict = gensim.corpora.Dictionary(documents=[[''],
                                                          question_tokens,
                                                          answer_tokens,
                                                          context_tokens]).token2id
        np_questions, np_answers, np_contexts, ids, np_as \
            = generate_numpy_features_from_squad_examples([example], vocab_dict)
        assert np_questions.shape == (1, config.MAX_QUESTION_WORDS)
        assert np_answers.shape == (1, config.MAX_ANSWER_WORDS)
        assert np_contexts.shape == (1, config.MAX_CONTEXT_WORDS)
        assert len(ids) == 1
        assert np_as.shape == (1, 2)
        #print(np_questions)
        #print(np_answers)
        #print(np_contexts)
        # Check that indices match up with question, answer and context
        for i in range(np_questions.shape[1]):
            if np_questions[0, i] != 0:
                assert np_questions[0, i] == vocab_dict[question_tokens[i]]
            else:
                assert i >= len(question_tokens)
        for i in range(np_answers.shape[1]):
            if np_answers[0, i] != 0:
                assert np_answers[0, i] == vocab_dict[answer_tokens[i]]
            else:
                assert i >= len(answer_tokens)
        for i in range(np_contexts.shape[1]):
            if np_contexts[0, i] != 0:
                assert np_contexts[0, i] == vocab_dict[context_tokens[i]]
            else:
                assert i >= len(answer_tokens)

    def test_generate_numpy_features_from_squad_examples_with_context_words(self):
        """Generate single example and test the output."""
        question = 'what is the meaning of life ?'
        answer = 'sloppy joe'
        context = 'life is an in deterministic sloppy joe'
        question_tokens = question.split()
        answer_tokens = answer.split()
        context_tokens = context.split()
        example = (question, answer, context, 1234, 5678)
        vocab_dict = gensim.corpora.Dictionary(documents=[[''],
                                                          question_tokens,
                                                          answer_tokens,
                                                          context_tokens]).token2id
        np_questions, np_answers, np_contexts, ids, np_as \
            = generate_numpy_features_from_squad_examples([example], vocab_dict,
                                                          answer_indices_from_context=True)
        assert np_questions.shape == (1, config.MAX_QUESTION_WORDS)
        assert np_answers.shape == (1, config.MAX_ANSWER_WORDS)
        assert np_contexts.shape == (1, config.MAX_CONTEXT_WORDS)
        assert len(ids) == 1
        assert np_as.shape == (1, 2)
        #print(np_questions)
        #print(np_answers)
        #print(np_contexts)
        # Check that indices match up with question, answer and context
        for i in range(np_questions.shape[1]):
            if np_questions[0, i] != 0:
                assert np_questions[0, i] == vocab_dict[question_tokens[i]]
            else:
                assert i >= len(question_tokens)
        for i in range(np_answers.shape[1]):
            if i < len(answer_tokens):
                pass
            else:
                assert np_answers[0, i] == 0
        assert np_answers.max() <= len(context_tokens)
        for i in range(np_contexts.shape[1]):
            if np_contexts[0, i] != 0:
                assert np_contexts[0, i] == vocab_dict[context_tokens[i]]
            else:
                assert i >= len(answer_tokens)

    def test_pipe(self):
        """Test functions in sequence to create numpy arrays."""
        [tk_paragraph] = tokenize_paragraphs([self.paragraph])
        qacs_tuples = convert_paragraphs_to_flat_format([tk_paragraph])
        m = len(qacs_tuples)
        vocab = generate_vocabulary_for_paragraphs([tk_paragraph])
        vocab_dict = vocab.token2id
        np_questions, np_answers, np_contexts, ids, np_as \
            = generate_numpy_features_from_squad_examples(qacs_tuples, vocab_dict)
        assert np_questions.shape == (m, config.MAX_QUESTION_WORDS)
        assert np_answers.shape == (m, config.MAX_ANSWER_WORDS)
        assert np_contexts.shape == (m, config.MAX_CONTEXT_WORDS)
        assert len(ids) == m
        assert np_as.shape == (m, 2)
        assert np.array_equal(np_contexts[0, :], np_contexts[1, :])
        assert np.array_equal(np_contexts[1, :], np_contexts[2, :])
        assert np.array_equal(np_questions[0, :], np_questions[1, :])
        #print(np_questions)
        #print(np_answers)
        #print(np_contexts)

    def test_construct_embeddings_for_vocab(self):
        vocab_dict = {'': 0, 'apples': 2, 'oranges': 1}
        np_embeddings = construct_embeddings_for_vocab(vocab_dict)
        assert np_embeddings.shape == (3, 300)
        #print(np_embeddings[0, :])
        #print(np_embeddings[0, :].shape)
        assert np.array_equal(np_embeddings[0, :], np.zeros((300,)))
        assert np.array_equal(np_embeddings[1, :], nlp('oranges').vector)
        assert np.array_equal(np_embeddings[2, :], nlp('apples').vector)

    def test_convert_numpy_array_to_strings(self):
        examples = convert_paragraphs_to_flat_format([self.paragraph])
        vocab_dict = generate_vocabulary_for_paragraphs([self.paragraph]).token2id
        #print(vocab_dict)
        np_questions, np_answers, np_contexts, ids, np_as = \
            generate_numpy_features_from_squad_examples(examples, vocab_dict)
        vocabulary = invert_dictionary(vocab_dict)
        #print(vocabulary)
        questions = convert_numpy_array_to_strings(np_questions, vocabulary)
        answers = convert_numpy_array_to_strings(np_answers, vocabulary)
        contexts = convert_numpy_array_to_strings(np_contexts, vocabulary)
        for i, each_example in enumerate(examples):
            assert questions[i] == each_example[0].lower()
            assert answers[i] == each_example[1].lower()
            assert contexts[i] == each_example[2].lower()

    def test_convert_numpy_array_answers_to_strings(self):
        examples = convert_paragraphs_to_flat_format([self.paragraph])
        contexts = [example[2] for example in examples]
        vocab_dict = generate_vocabulary_for_paragraphs([self.paragraph]).token2id
        #print(vocab_dict)
        np_questions, np_answers, np_contexts, ids, np_as = \
            generate_numpy_features_from_squad_examples(examples, vocab_dict, answer_indices_from_context=True)
        vocabulary = invert_dictionary(vocab_dict)
        #print(vocabulary)
        questions = convert_numpy_array_to_strings(np_questions, vocabulary)
        answers = convert_numpy_array_answers_to_strings(np_answers, contexts)
        contexts = convert_numpy_array_to_strings(np_contexts, vocabulary)
        for i, each_example in enumerate(examples):
            assert questions[i] == each_example[0].lower()
            assert answers[i] == each_example[1].lower()
            assert contexts[i] == each_example[2].lower()

    def test_convert_numpy_array_answers_to_strings_answer_is_span(self):
        examples = convert_paragraphs_to_flat_format([self.paragraph])
        contexts = [example[2] for example in examples]
        vocab_dict = generate_vocabulary_for_paragraphs([self.paragraph]).token2id
        #print(vocab_dict)
        np_questions, np_answers, np_contexts, ids, np_as = \
            generate_numpy_features_from_squad_examples(examples, vocab_dict, answer_indices_from_context=True,
                                                        answer_is_span=True)
        vocabulary = invert_dictionary(vocab_dict)
        #print(vocabulary)
        questions = convert_numpy_array_to_strings(np_questions, vocabulary)
        answers = convert_numpy_array_answers_to_strings(np_answers, contexts, answer_is_span=True)
        contexts = convert_numpy_array_to_strings(np_contexts, vocabulary)
        for i, each_example in enumerate(examples):
            assert questions[i] == each_example[0].lower()
            assert answers[i] == each_example[1].lower()
            assert contexts[i] == each_example[2].lower()

    def test_compute_span_accuracy(self):
        np_labels = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        np_predictions = np.array([[0, 0], [2, 2], [4, 5], [6, 7]])
        accuracy = compute_multi_label_accuracy(np_labels, np_predictions)
        assert accuracy == 0.5
        labels_same_accuracy = compute_multi_label_accuracy(np_labels, np_labels)
        predictions_same_accuracy = compute_multi_label_accuracy(np_predictions, np_predictions)
        assert labels_same_accuracy == 1.0
        assert predictions_same_accuracy == 1.0
        labels_zeros_accuracy = compute_multi_label_accuracy(np.zeros([4, 2]), np_labels)
        assert labels_zeros_accuracy == 0.0

    def test_compute_mask_accuracy(self):
        np_first = np.array([[1, 2, 3, 0, 0]])
        np_second = np.array([[1, 2, 4, 0, 0]])
        np_mask = np.array([[1, 1, 1, .2, 0]])
        accuracy, word_accuracy = compute_mask_accuracy(np_first, np_second, np_mask)
        assert word_accuracy == 0.6875
        assert accuracy == 0

        np_first = np.array([[1, 2, 3, 0, 0], [1, 7, 4, 0, 1]])
        np_second = np.array([[1, 2, 4, 0, 0], [1, 7, 4, 0, 0]])
        np_mask = np.array([[1, 1, 1, .2, 0], [1, 1, 1, .2, 0]])
        accuracy, word_accuracy = compute_mask_accuracy(np_first, np_second, np_mask)
        print(word_accuracy)
        assert word_accuracy == 0.84375
        assert accuracy == 0.5

        same_accuracy, same_word_accuracy = compute_mask_accuracy(np_first, np_first, np_mask)
        assert same_accuracy == 1.0
        assert same_word_accuracy == 1.0

    def test_remove_excess_spaces_from_paragraphs(self):
        space_filled_paragraph = {'context': 'lots  of  spaces ', 'qas':
                                  [{'question': ' question space  space space',
                                    'id': 55,
                                    'answers': [{'text': 'lots  of', 'answer_start': 10}]}]}
        [clean_paragraph] = remove_excess_spaces_from_paragraphs([space_filled_paragraph])
        assert clean_paragraph['context'] == 'lots of spaces'
        assert clean_paragraph['qas'][0]['question'] == 'question space space space'
        assert clean_paragraph['qas'][0]['answers'][0]['text'] == 'lots of'






