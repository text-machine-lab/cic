"""Script for responding to user message using QA model and latent chat model."""
import optparse
import pickle

import gensim
import os
import tensorflow as tf
from cic.chat_bots import latent_chat
from cic.qa import match_lstm, squad_tools as sdt

from cic import config
from cic.chat_bots import chat_model

parser = optparse.OptionParser()
parser.add_option('-s', '--save_dir', dest="save_dir", default=config.LATENT_CHAT_MODEL_SAVE_DIR, help='specify save directory for training and restoring')
parser.add_option('-a', '--ae_save_dir', dest="auto_encoder_save_dir", default=config.AUTO_ENCODER_MODEL_SAVE_DIR, help='specify save directory for auto-encoder model')

(options, args) = parser.parse_args()

sdt.initialize_nlp()
with tf.Graph().as_default() as latent_chat_model_graph:
    vocab_dict_filename = os.path.join(options.auto_encoder_save_dir, 'vocab_dict.pkl')
    chat_vocab_dict = pickle.load(open(vocab_dict_filename, 'rb'))
    chat_vocabulary = sdt.invert_dictionary(chat_vocab_dict)
    lcm = latent_chat.LatentChatModel(len(chat_vocab_dict), latent_chat.LEARNING_RATE,
                                      options.save_dir,
                                      ae_save_dir=options.auto_encoder_save_dir,
                                      restore_from_save=True)

with tf.Graph().as_default() as qa_model_graph:
    qa_model = match_lstm.LSTMBaselineModel(match_lstm.RNN_HIDDEN_DIM, 0.0,
                                            save_dir=config.BASELINE_MODEL_SAVE_DIR,
                                            restore_from_save=True)

context = input('Enter context: ')
while True:
    message = input('Message: ')

    # Latent chat model
    response = lcm.predict_string(message, sdt.nlp, chat_vocab_dict, chat_vocabulary)

    # QA model
    message_tokenize = sdt.nlp.tokenizer(message)
    tk_message = ' '.join([str(token) for token in message_tokenize]).lower()
    context_tokenize = sdt.nlp.tokenizer(context)
    tk_context = ' '.join([str(token) for token in context_tokenize]).lower()
    vocab_dict = gensim.corpora.Dictionary(documents=[tk_message.split(), tk_context.split()]).token2id
    np_embeddings = sdt.construct_embeddings_for_vocab(vocab_dict)
    np_question = chat_model.construct_numpy_from_messages([tk_message.split()], vocab_dict, config.MAX_QUESTION_WORDS)
    np_context = chat_model.construct_numpy_from_messages([tk_context.split()], vocab_dict, config.MAX_CONTEXT_WORDS)
    np_prediction, np_probability = qa_model.predict_on_examples(np_embeddings, np_question, np_context, 1)
    answer = ' '.join(sdt.convert_numpy_array_answers_to_strings(np_prediction, [context], zero_stop_token=True))
    print('Answer: %s' % answer)
    print('Response: %s' % response)
    #print(tk_message)
    #print(vocab_dict)
