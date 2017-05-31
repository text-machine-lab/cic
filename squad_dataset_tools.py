"""Copyright 2017 David Donahue. LSTM baseline for SQuAD dataset. Reads question with LSTM.
Reads passage with LSTM. Outputs answer with LSTM."""
import config
import json
import unittest2
import pprint
import gensim
import spacy
import numpy as np


nlp = spacy.load('en_vectors_glove_md')  # python -m spacy download en
#nlp = spacy.load('en')


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


def convert_numpy_array_to_strings(np_examples, vocabulary):
    """Converts a numpy array of indices into a list of strings.

    np_examples - m x n numpy array of ints, where m is the number of
    strings encoded by indices, and n is the max length of each string
    vocab_dict - where vocab_dict[index] gives a word in the vocabulary
    with that index

    Returns: a list of strings, where each string is constructed from
    indices in the array as they appear in the vocabulary."""
    m = np_examples.shape[0]
    n = np_examples.shape[1]
    examples = []
    for i in range(m):
        each_example = ''
        for j in range(n):
            word_index = np_examples[i][j]
            word = vocabulary[word_index]
            if j > 0 and word != '':
                each_example += ' '
            each_example += word
        examples.append(each_example)
    return examples


def convert_numpy_array_answers_to_strings(np_answers, contexts):
    """Converts a numpy array of answer indices into a list of strings.
    Indices are taken from the context of each answer.

    np_examples - m x n numpy array of ints, where m is the number of
    strings encoded by indices, and n is the max length of each string
    vocab_dict - where vocab_dict[index] gives a word in the vocabulary
    with that index
    contexts - list of contexts, one for each answer

    Returns: a list of strings, where each string is constructed from
    indices in the array as they appear in the context."""
    m = np_answers.shape[0]
    n = np_answers.shape[1]
    assert m == len(contexts)
    answer_strings = []
    for i in range(m):
        answer_string = ''
        context_tokens = contexts[i].split()
        for j in range(n):
            if np_answers[i, j] != 0:
                context_index = np_answers[i, j] - 1  # was shifted for index 0 -> ''
                if context_index < len(context_tokens):
                    if j > 0:
                        answer_string += ' '
                    answer_string += context_tokens[context_index]
        answer_strings.append(answer_string)
    return answer_strings


def invert_dictionary(dictionary):
    inv_dictionary = {v: k for k, v in dictionary.items()}
    return inv_dictionary


def construct_embeddings_for_vocab(vocab_dict):
    """Creates matrix to hold embeddings for words in vocab, ordered by index in vocabulary.
    Intended to be used as lookup embedding tensor.

    Returns: embedding matrix for all words in the vocab with an associated embedding."""
    m = len(vocab_dict.keys())
    np_embeddings = np.zeros([m, config.SPACY_GLOVE_EMB_SIZE])
    for each_word in vocab_dict:
        index = vocab_dict[each_word]
        embedding = nlp(each_word)
        #print(embedding.vector)
        np_embeddings[index, :] = embedding.vector

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


def generate_vocabulary_for_paragraphs(paragraphs):
    """Generates a token-based vocabulary using gensim for all words in contexts, questions and answers of
    'paragraphs'. Converts all tokens to lowercase

    paragraphs - a list of {'context', 'qas'} dictionaries where context is the paragraph and qas is a list of
    {'answers', 'question', 'id'} tuples
    prune_at - maximum size of vocabulary, infrequent words will be pruned after this level

    Returns: gensim.corpora.Dictionary object containing vocabulary. Can view dictionary mapping with self.token2id."""
    documents = []
    vocabulary = gensim.corpora.Dictionary()
    for each_paragraph in paragraphs:
        context = each_paragraph['context']
        documents.append(context.lower().split())
        for each_qas in each_paragraph['qas']:
            question = each_qas['question']
            documents.append(question.lower().split())
            for each_answer in each_qas['answers']:
                answer = each_answer['text']
                documents.append(answer.lower().split())
    vocabulary.add_documents([['']])
    vocabulary.add_documents(documents)

    return vocabulary


def generate_numpy_features_from_squad_examples(examples, vocab_dict,
                                                max_question_words=config.MAX_QUESTION_WORDS,
                                                max_answer_words=config.MAX_ANSWER_WORDS,
                                                max_context_words=config.MAX_CONTEXT_WORDS,
                                                answer_indices_from_context=False):
    """Uses a list of squad QA examples to generate features for a QA model. Features are numpy arrays for
    questions, answers, and contexts, where each word is represented as an index in a vocabulary. Answer
    indices can either be taken from general vocabulary or context vocabulary.

    examples - list of (question, answer, context, id, answer_start) tuples
    vocab_dict - mapping from words to indices in vocabulary
    max_question_words - max length of vector containing question word ids
    max_answer_words - max length of vector containing answer word ids
    max_context_words - max length of vector containing context word ids
    answer_indices_from_context - if true, words in each answer will be represented as the indices of matching words
    in the context (+1 for each index, leaving room for index 0 -> '').
    Otherwise, words in each answer will be represented as indices from vocabulary vocab_dict

    Returns: where m is the number of examples, an m x max_question_words array for questions, an m x max_answer_words
    array for answers, and an m x max_context_words array for context, an m-dimensional vector
    for question ids and an m by 2 vector for answer_starts/ends. Ie. returns (np_questions, np_answers, np_contexts, np_ids, np_as)."""
    m = len(examples)
    np_questions = np.zeros([m, max_question_words], dtype=int)
    np_answers = np.zeros([m, max_answer_words], dtype=int)
    np_contexts = np.zeros([m, max_context_words], dtype=int)
    ids = []
    np_as = np.zeros([m, 2])

    for i, each_example in enumerate(examples):
        question, answer, context, id, answer_start = each_example
        question_tokens = question.lower().split()
        answer_tokens = answer.lower().split()
        context_tokens = context.lower().split()
        # assert len(question_tokens) <= max_question_words
        # assert len(answer_tokens) <= max_answer_words
        # assert len(context_tokens) <= max_context_words
        answer_end = answer_start + len(answer_tokens)
        for j, each_token in enumerate(question_tokens):
            if j < max_question_words and each_token in vocab_dict:
                np_questions[i, j] = vocab_dict[each_token]
        for j, each_token in enumerate(context_tokens):
            if j < max_context_words and each_token in vocab_dict:
                np_contexts[i, j] = vocab_dict[each_token]

        if answer_indices_from_context:
            # for j, each_token in enumerate(answer_tokens):
            #     if j < max_answer_words and each_token in context_tokens:
            #         answer_word_index = context_tokens.index(each_token) + 1  # index 0 -> ''
            #
            #         if answer_word_index < config.MAX_CONTEXT_WORDS:
            #             np_answers[i, j] = answer_word_index
            for context_index, each_context_token in enumerate(context_tokens):
                answer_starts_here = True
                for answer_index, each_answer_token in enumerate(answer_tokens):
                    if context_index + answer_index >= len(context_tokens) or context_tokens[context_index + answer_index] != each_answer_token:
                        answer_starts_here = False
                if answer_starts_here:
                    for answer_index in range(len(answer_tokens)):
                        if answer_index < config.MAX_ANSWER_WORDS and context_index + answer_index < config.MAX_CONTEXT_WORDS:
                            np_answers[i, answer_index] = context_index + answer_index + 1  # index 0 -> ''
        else:
            for j, each_token in enumerate(answer_tokens):
                if j < max_answer_words and each_token in vocab_dict:
                    np_answers[i, j] = vocab_dict[each_token] # index 0 -> ''
        ids.append(id)
        np_as[i, 0] = answer_start
        np_as[i, 1] = answer_end

    return np_questions, np_answers, np_contexts, ids, np_as


class LSTM_Baseline_Test(unittest2.TestCase):
    def setUp(self):
        self.answer = {'text': 'once upon', 'answer_start': 69}
        self.answer2 = {'text': 'a time there was', 'answer_start': 123}
        self.answer3 = {'text': 'a useless paragraph.', 'answer_start': 0}
        self.qas1 = {'question': 'can birds fly?', 'id': 54321, 'answers': [self.answer, self.answer2]}
        self.qas2 = {'question': 'what is the meaning of life?', 'id': 5454, 'answers': [self.answer3]}
        self.paragraph = {'context': 'once upon a time there was a useless paragraph. The end.', 'qas': [self.qas1, self.qas2]}

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
        m = len(vocab_dict.keys())
        np_embeddings = construct_embeddings_for_vocab(vocab_dict)
        assert np_embeddings.shape == (3, 300)
        print(np_embeddings[0, :])
        print(np_embeddings[0, :].shape)
        assert np.array_equal(np_embeddings[0, :], np.zeros((300,)))
        assert np.array_equal(np_embeddings[1, :], nlp('oranges').vector)
        assert np.array_equal(np_embeddings[2, :], nlp('apples').vector)

    def test_convert_numpy_array_to_strings(self):
        examples = convert_paragraphs_to_flat_format([self.paragraph])
        vocab_dict = generate_vocabulary_for_paragraphs([self.paragraph]).token2id
        print(vocab_dict)
        np_questions, np_answers, np_contexts, ids, np_as = \
            generate_numpy_features_from_squad_examples(examples, vocab_dict)
        vocabulary = invert_dictionary(vocab_dict)
        print(vocabulary)
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
        print(vocab_dict)
        np_questions, np_answers, np_contexts, ids, np_as = \
            generate_numpy_features_from_squad_examples(examples, vocab_dict, answer_indices_from_context=True)
        vocabulary = invert_dictionary(vocab_dict)
        print(vocabulary)
        questions = convert_numpy_array_to_strings(np_questions, vocabulary)
        answers = convert_numpy_array_answers_to_strings(np_answers, contexts)
        contexts = convert_numpy_array_to_strings(np_contexts, vocabulary)
        for i, each_example in enumerate(examples):
            assert questions[i] == each_example[0].lower()
            assert answers[i] == each_example[1].lower()
            assert contexts[i] == each_example[2].lower()







