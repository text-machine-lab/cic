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

    Returns: gensim.corpora.Dictionary object containing vocabulary. Can view dictionary mapping with self.token2id."""
    documents = []
    for each_paragraph in paragraphs:
        context = each_paragraph['context']
        documents.append(context.lower().split())
        for each_qas in each_paragraph['qas']:
            question = each_qas['question']
            documents.append(question.lower().split())
            for each_answer in each_qas['answers']:
                answer = each_answer['text']
                documents.append(answer.lower().split())
    return gensim.corpora.Dictionary(documents)


def generate_numpy_features_from_squad_examples(examples, vocab_dict,
                                                max_question_words=config.MAX_QUESTION_WORDS,
                                                max_answer_words=config.MAX_ANSWER_WORDS,
                                                max_context_words=config.MAX_CONTEXT_WORDS):
    """Uses a list of squad QA examples to generate features for a QA model.

    examples - list of (question, answer, context, id, answer_start) tuples
    word_to_index - mapping from words to indices in vocabulary
    max_question_words - max length of vector containing question word ids
    max_answer_words - max length of vector containing answer word ids
    max_context_words - max length of vector containing context word ids

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
        question_tokens = question.split()
        answer_tokens = answer.split()
        context_tokens = context.split()
        # assert len(question_tokens) <= max_question_words
        # assert len(answer_tokens) <= max_answer_words
        # assert len(context_tokens) <= max_context_words
        answer_end = answer_start + len(answer_tokens)
        for j, each_token in enumerate(question_tokens):
            if j < max_question_words:
                np_questions[i, j] = vocab_dict[each_token]
        for j, each_token in enumerate(answer_tokens):
            if j < max_answer_words:
                np_answers[i, j] = vocab_dict[each_token]
        for j, each_token in enumerate(context_tokens):
            if j < max_context_words:
                np_contexts[i, j] = vocab_dict[each_token]
        ids.append(id)
        np_as[i, 0] = answer_start
        np_as[i, 1] = answer_end

    return np_questions, np_answers, np_contexts, ids, np_as


class LSTM_Baseline_Test(unittest2.TestCase):
    def setUp(self):
        self.answer = {'text': 'hello world', 'answer_start': 69}
        self.answer2 = {'text': 'what are you talking about', 'answer_start': 123}
        self.answer3 = {'text': 'when did that van get there?', 'answer_start': 0}
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
        assert ('can birds fly?', 'hello world', 'once upon a time there was a useless paragraph. The end.', 54321, 69) in qacs_tuples
        assert ('what is the meaning of life?', 'when did that van get there?', 'once upon a time there was a useless paragraph. The end.', 5454, 0) in qacs_tuples
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






