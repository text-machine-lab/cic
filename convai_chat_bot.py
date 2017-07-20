"""Script for responding to user message using QA model and latent chat model."""
import latent_chat_func
import baseline_model_func
import sys
import config
import pickle

vocab_dict = pickle.load(open(config.AUTO_ENCODER_VOCAB_DICT, 'rb'))
lcm = latent_chat_func.LatentChatModel(len(vocab_dict), latent_chat_func.LEARNING_RATE,
                                       config.LATENT_CHAT_MODEL_SAVE_DIR,
                                       restore_from_save=True)

qa_model = baseline_model_func.LSTMBaselineModel(baseline_model_func.RNN_HIDDEN_DIM, 0.0,
                                                 save_dir=config.BASELINE_MODEL_SAVE_DIR,
                                                 restore_from_save=True)

for line in sys.stdin:
    
    print(line)

