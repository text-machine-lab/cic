"""Train the Generic Model autoencoder on the UKWac dataset. Evaluate its
accuracy."""
import os

import cic.config
from cic.autoencoders.gm_auto_encoder import AutoEncoder
from cic.gemtk_datasets.uk_wac_dataset import UKWacDataset
from cic.gemtk_datasets.book_corpus_dataset import TorontoBookCorpus
from cic.gemtk_datasets.string_dataset import convert_numpy_array_to_strings
from sacred import Experiment
import numpy as np
import pickle
ex = Experiment('ukwac')

@ex.config
def config():
    max_sentence_length = 20
    min_sentence_length = 5
    save_dir = cic.config.GM_AE_SAVE_DIR
    print('Save directory: %s' % save_dir)
    restore_from_save = True
    num_epochs = 0
    regenerate_dataset = False # If problem with dataset, try this first
    rnn_size = 600
    learning_rate = 0.0005
    max_number_of_sentences = 2000000
    train_test_split=0.99
    split_seed = 'seed'
    i_erased_vocabulary_by_accident = True

@ex.automain
def main(max_sentence_length, save_dir,
         restore_from_save, num_epochs,
         regenerate_dataset, rnn_size, learning_rate,
         max_number_of_sentences, train_test_split,
         split_seed, min_sentence_length,
         i_erased_vocabulary_by_accident):

    # Load UKWac dataset
    print('Loading dataset...')

    #ukwac_path = '/data2/arogers/Corpora/En/UkWac/Plain-txt/ukwac_subset_100M.txt'
    #result_path = os.path.join(cic.config.DATA_DIR, 'ukwac')

    # ukwac = UKWacDataset(ukwac_path, result_save_path=result_path, max_length=max_sentence_length,
    #                      regenerate=regenerate_dataset, max_number_of_sentences=max_number_of_sentences,
    #                      min_length=min_sentence_length)

    vocab = None
    if i_erased_vocabulary_by_accident:
        print('Loading vocabulary from save...')
        vocab_path = os.path.join(cic.config.BOOK_CORPUS_RESULT, 'vocab.pkl')
        with open(vocab_path, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)

    tbc = TorontoBookCorpus(20, result_path=cic.config.BOOK_CORPUS_RESULT,
                            min_length=5, max_num_s=max_number_of_sentences, keep_unk_sentences=False,
                            vocab_min_freq=5, vocab=vocab, regenerate=regenerate_dataset)

    print('Len UKWac dataset: %s' % len(tbc))

    # # Check length limits
    # for sentence in ukwac.messages:
    #     tokens = sentence.split()
    #     if not (len(tokens)+1 >= min_sentence_length and len(tokens) <= max_sentence_length):
    #         print('Out of bounds: %s' % str(tokens))

    print('Dividing dataset into train/validation split...')

    train_tbc, val_tbc = tbc.split(train_test_split, seed=0.9)

    print('Len training set: %s' % len(train_tbc))
    print('Len validation set: %s' % len(val_tbc))

    token_to_id = tbc.vocab
    id_to_token = {tbc.vocab[k]:k for k in tbc.vocab}

    print('Len vocabulary: %s' % len(token_to_id))

    # Create autoencoder
    print('Constructing autoencoder...')

    autoencoder = AutoEncoder(len(token_to_id), tensorboard_name='gmae', save_dir=save_dir,
                              restore_from_save=restore_from_save, max_len=max_sentence_length, rnn_size=rnn_size)

    # Train autoencoder
    if num_epochs > 0:
        print('Training autoencoder...')

        autoencoder.train(train_tbc, output_tensor_names=['train_prediction'],
                          parameter_dict={'keep prob': 0.9, 'learning rate': learning_rate},
                          num_epochs=num_epochs, batch_size=20, verbose=True)

    # Calculate train accuracy
    # print('Calculating training accuracy...')
    #
    # results = autoencoder.predict(train_tbc, output_tensor_names=['train_prediction'])
    # np_predictions = results['train_prediction']
    # #predictions = ukwac.convert_numpy_to_strings(np_predictions)
    # predictions = convert_numpy_array_to_strings(np_predictions, id_to_token,
    #                                           tbc.stop_token,
    #                                           keep_stop_token=False)
    #
    # total_reconstructions = len(predictions)
    # correct_reconstructions = 0
    #
    # print('Len predictions: %s' % len(predictions))
    #
    # for index in range(len(predictions)):
    #     each_prediction = predictions[index]
    #     each_np_prediction = np_predictions[index,:]
    #
    #     each_np_original = np.reshape(tbc[train_tbc.indices[index]]['message'], newshape=[1, -1])
    #
    #     each_original = convert_numpy_array_to_strings(each_np_original, id_to_token,
    #                                                    tbc.stop_token,
    #                                                    keep_stop_token=False)[0]
    #
    #     if index < 10:
    #
    #         print('Original: %s' % each_original)
    #         print('Original numpy: %s' % str(each_np_original))
    #         print('Reconstruction: %s' % each_prediction)
    #         print('Reconstruction numpy: %s' % str(each_np_prediction))
    #
    #
    #     if each_prediction == each_original:
    #         correct_reconstructions += 1
    #
    # print('Training accuracy: %s' % (correct_reconstructions / total_reconstructions))
    # print()
    # Calculate validation accuracy
    print('Calculating validation accuracy...')

    results = autoencoder.predict(val_tbc, output_tensor_names=['train_prediction'])
    np_predictions = results['train_prediction']

    predictions = convert_numpy_array_to_strings(np_predictions, id_to_token,
                                                 tbc.stop_token,
                                                 keep_stop_token=False)


    total_reconstructions = len(predictions)
    correct_reconstructions = 0

    for index in range(len(predictions)):
        each_prediction = predictions[index]
        each_np_original = np.reshape(tbc[val_tbc.indices[index]]['message'], newshape=[1,-1])

        each_original = convert_numpy_array_to_strings(each_np_original, id_to_token,
                                                    tbc.stop_token,
                                                    keep_stop_token=False)[0]

        if index < 10:
            print('Reconstruction: %s' % each_prediction)
            print('Original: %s' % each_original)

        if each_prediction == each_original:
            correct_reconstructions += 1

    print('Validation accuracy: %s' % (correct_reconstructions / total_reconstructions))

