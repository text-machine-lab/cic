"""Copyright 2017 David Donahue. Stores global paths and constants."""
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
print(__file__)
CORNELL_MOVIE_DIR = os.path.join(DATA_DIR, 'cornell movie-dialogs corpus/')

CORNELL_MOVIE_LINES_FILE = os.path.join(CORNELL_MOVIE_DIR, 'movie_lines.txt')
CORNELL_MOVIE_CONVERSATIONS_FILE = os.path.join(CORNELL_MOVIE_DIR, 'movie_conversations.txt')
MS_MARCO_TRAIN_SET = os.path.join(DATA_DIR, 'ms_marco/train_v1.1.json')
SQUAD_TRAIN_SET = os.path.join(DATA_DIR, 'squad/train-v1.1.json')
BASELINE_MODEL_SAVE_DIR = os.path.join(DATA_DIR, 'baseline_models/')
CHAT_MODEL_SAVE_DIR = os.path.join(DATA_DIR, 'chat_models/')
AUTO_ENCODER_MODEL_SAVE_DIR = os.path.join(DATA_DIR, 'auto_encoder_models/')
AUTO_ENCODER_VOCAB_DICT = os.path.join(AUTO_ENCODER_MODEL_SAVE_DIR, 'vocab_dict.pkl')
LATENT_CHAT_MODEL_SAVE_DIR = os.path.join(DATA_DIR, 'latent_chat_models/')
GLOVE_200_FILE = os.path.join(DATA_DIR, 'glove.twitter.27B/glove.twitter.27B.200d.txt')
VALIDATION_PREDICTIONS_FILE = os.path.join(DATA_DIR, 'predictions.pkl')
SUBMISSION_PREDICTIONS_FILE = os.path.join(DATA_DIR, 'final_predictions.json')
REDDIT_CRAWLER_CREDENTIALS = os.path.join(DATA_DIR, 'creds.txt')
REDDIT_COMMENTS_DUMP = os.path.join(DATA_DIR, 'reddit_comments_dump')
LATENT_CHAT_PRE_PROCESSING_DUMP = os.path.join(DATA_DIR, 'latent_chat_preproc_dump.pkl')

# CONSTANTS
MAX_QUESTION_WORDS = 10
MAX_ANSWER_WORDS = 10
MAX_CONTEXT_WORDS = 200
GLOVE_EMB_SIZE = 300
