import unittest2

from gemtk_datasets.cornell_movie_dialogues_dataset import *


class CornellMovieDialoguesTest(unittest2.TestCase):
    def test_empty(self):
        pass

    def test_reconstruct(self):
        """Confirm that all strings converted into numpy form can be converted back to their original string form.
        Excludes strings that go beyond max length limit (these would get cut off)."""
        cmd_dataset = CornellMovieDialoguesDataset(config.CORNELL_MOVIE_LINES_FILE)

        token_to_id, id_to_token = cmd_dataset.get_vocabulary()
        assert len(token_to_id) == len(id_to_token)
        nlp = spacy.load('en_core_web_sm')
        m = len(cmd_dataset)
        assert m == len(cmd_dataset.formatted_and_filtered_strings)
        for index in range(m):
            each_example = cmd_dataset[index]
            each_np_message = np.reshape(each_example['message'], newshape=(-1, 30))
            each_reconstructed_message = sdt.convert_numpy_array_to_strings(each_np_message, id_to_token,
                                                                            stop_token=cmd_dataset.stop_token,
                                                                            keep_stop_token=True)[0]
            each_message = cmd_dataset.formatted_and_filtered_strings[index]
            if len(cmd_dataset.formatted_and_filtered_strings[index].split()) <= 30:
                if each_message != each_reconstructed_message:
                    print(each_message)
                    print(each_reconstructed_message)
                    exit()

    def test_len(self):
        cmd_dataset = CornellMovieDialoguesDataset(config.CORNELL_MOVIE_LINES_FILE, max_message_length=10)

        for index in range(cmd_dataset.np_messages.shape[0]):
            assert cmd_dataset.np_messages[index, -1] == cmd_dataset.token_to_id[cmd_dataset.stop_token] or \
                   cmd_dataset.np_messages[index, -1] == cmd_dataset.token_to_id['']