import arcadian.dataset
import spacy
import cic.models.old_chat_model

class CornellMovieConversationDataset(arcadian.dataset.Dataset):
    def __init__(self, max_s_len, reverse_inputs=False, seed='seed',
                 stop_token='<STOP>', save_dir=None, max_vocab_len=10000, regenerate=False):
        self.nlp = spacy.load('en')

        self.stop_token = stop_token
        self.max_s_len = max_s_len

        self.examples, self.messages, self.responses, self.vocab, self.inv_vocab\
            = cic.models.old_chat_model.preprocess_all_cornell_conversations(self.nlp, reverse_inputs=reverse_inputs,
                                                                             verbose=True,
                                                                             stop_token=stop_token,
                                                                             keep_duplicates=True, seed=seed,
                                                                             max_message_length=max_s_len,
                                                                             save_dir=save_dir,
                                                                             regen=regenerate,
                                                                             max_vocab_len=max_vocab_len)

        assert self.messages.shape[0] == self.responses.shape[0]

    def __getitem__(self, index):
        return {'message': self.messages[index, :],
                'response': self.responses[index, :]}

    def __len__(self):
        return self.messages.shape[0]