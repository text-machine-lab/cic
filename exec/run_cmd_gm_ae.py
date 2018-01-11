"""Trains an autoencoder (with arcadian integration) on the Cornell Movie Dialogues dataset (with arcadian integration)."""

# EXECUTION ############################################################################################################
import pickle

import os
import cic.utils.squad_tools as sdt

from cic import config
from cic.models.gm_autoencoder import AutoEncoder
from cic.datasets import cornell_movie_dialogues as cmd

RESTORE_FROM_SAVE = False
SAVE_DIR = './data/autoencoder/first/'

if __name__ == '__main__':
    saved_token_to_id = None
    if RESTORE_FROM_SAVE:
        # Reuse vocabulary when restoring from save
        saved_token_to_id = pickle.load(open(os.path.join(SAVE_DIR, 'vocabulary.pkl'), 'rb'))

    cmd_dataset = cmd.CornellMovieDialoguesDataset(max_s_len=10, token_to_id=saved_token_to_id,
                                                   cornell_movie_lines_file=config.CORNELL_MOVIE_LINES_FILE)

    train_cmd, val_cmd = cmd_dataset.split(fraction=0.9, seed='hello world')

    print('Number of training examples: %s' % len(train_cmd))
    print('Number of validation examples: %s' % len(val_cmd))

    token_to_id, id_to_token = cmd_dataset.get_vocabulary()

    if RESTORE_FROM_SAVE:
        assert saved_token_to_id == token_to_id

    for i in range(10):
        print(id_to_token[i])

    # Save vocabulary
    pickle.dump(token_to_id, open(os.path.join(SAVE_DIR, 'vocabulary.pkl'), 'wb'))

    autoencoder = AutoEncoder(len(token_to_id), tensorboard_name='gmae', save_dir=SAVE_DIR,
                              restore_from_save=RESTORE_FROM_SAVE, max_len=10)

    # autoencoder.train(train_cmd, output_tensor_names=['train_prediction'],
    #                   parameter_dict={'keep prob': 0.9, 'learning rate': .0005},
    #                   num_epochs=100, batch_size=20, verbose=True)

    def calculate_train_accuracy():
        predictions = autoencoder.predict(train_cmd, output_tensor_names=['prediction'])['prediction']

        # Here, I need to convert predictions back to English and print
        reconstructed_messages = sdt.convert_numpy_array_to_strings(predictions, id_to_token,
                                                                    stop_token=cmd_dataset.stop_token,
                                                                    keep_stop_token=True)

        for i in range(10):
            print(' '.join(cmd_dataset.messages[train_cmd.indices[i]]) + " | " + reconstructed_messages[i])

        num_train_correct = 0
        for i in range(len(reconstructed_messages)):
            original_message = ' '.join(cmd_dataset.messages[train_cmd.indices[i]])
            if original_message == reconstructed_messages[i]:
                num_train_correct += 1

        print('Train EM accuracy: %s' % (num_train_correct / len(reconstructed_messages)))


    def predict_using_autoencoder_and_calculate_accuracy():

        val_predictions = autoencoder.predict(val_cmd, output_tensor_names=['prediction'])['prediction']

        val_reconstructed_messages = sdt.convert_numpy_array_to_strings(val_predictions, id_to_token,
                                                                        stop_token=cmd_dataset.stop_token,
                                                                        keep_stop_token=True)
        for i in range(10):
            print(' '.join(cmd_dataset.messages[val_cmd.indices[i]]) + " | " + val_reconstructed_messages[i])

        num_val_correct = 0
        for i in range(len(val_reconstructed_messages)):
            original_message = ' '.join(cmd_dataset.messages[val_cmd.indices[i]])
            if original_message == val_reconstructed_messages[i]:
                num_val_correct += 1

        print('Validation EM accuracy: %s' % (num_val_correct / len(val_reconstructed_messages)))


    def input_arbitrary_messages_into_autoencoder():
        print('Testing the autoencoder...')
        # Test autoencoder using stdin
        while True:
            message = input('Message: ')
            np_message = cmd_dataset.convert_strings_to_numpy([message])
            print(np_message)
            np_code = autoencoder.encode(np_message)
            print(np_code[:10])
            np_message_reconstruct = \
                autoencoder.predict(np_message, output_tensor_names=['prediction'])['prediction']
            message_reconstruct = cmd_dataset.convert_numpy_to_strings(np_message_reconstruct)[0]
            print('Reconstruct: %s' % message_reconstruct)

    calculate_train_accuracy()
    print()
    predict_using_autoencoder_and_calculate_accuracy()
    input_arbitrary_messages_into_autoencoder()

