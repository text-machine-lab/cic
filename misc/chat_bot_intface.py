"""An importable file that allows for automatic construction of
the new chat bot seq-to-seq model."""
from cic.datasets.cornell_movie_conversation import CornellMovieConversationDataset
from cic.models.seq_to_seq import Seq2Seq
from cic.exec.run_chat_model import generate_response_from_model
import cic.paths
import os

max_s_len = 10
emb_size = 200
rnn_size = 200
n = 5
save_dir = os.path.join(cic.paths.DATA_DIR, 'chat_model/')
cornell_dir = os.path.join(cic.paths.DATA_DIR, 'cornell_convos/')
max_vocab_len = 10000

ds = CornellMovieConversationDataset(max_s_len, reverse_inputs=False, seed='seed',
                                     save_dir=cornell_dir, max_vocab_len=max_vocab_len,
                                     regenerate=False)

model = Seq2Seq(max_s_len, len(ds.vocab), emb_size, rnn_size,
                save_dir=save_dir, restore=True, tensorboard_name='chat')

reverse_vocab = {ds.vocab[k]: k for k in ds.vocab}


def generate_response(msg):

    response = generate_response_from_model(msg, ds, model, n, reverse_vocab)

    return response
