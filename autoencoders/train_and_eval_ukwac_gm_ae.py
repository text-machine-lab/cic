"""Train the Generic Model autoencoder on the UKWac dataset. Evaluate its
accuracy."""
import os

import cic.config
from cic.autoencoders.gm_auto_encoder import AutoEncoder
from cic.gemtk_datasets.uk_wac_dataset import UKWacDataset
from sacred import Experiment
ex = Experiment('ukwac')


@ex.config
def config():
    max_sentence_length = 10
    save_dir = os.path.join(cic.config.DATA_DIR, 'ukwac_autoencoder2')
    print('Save directory: %s' % save_dir)
    restore_from_save = True
    num_epochs = 0
    regenerate_dataset = False  # If problem with dataset, try this first
    rnn_size = 600
    learning_rate = 0.0005


@ex.automain
def main(max_sentence_length, save_dir,
         restore_from_save, num_epochs,
         regenerate_dataset, rnn_size, learning_rate):
    # Load UKWac dataset
    print('Loading dataset...')
    ukwac_path = '/data2/arogers/Corpora/En/UkWac/Plain-txt/ukwac_subset_100M.txt'
    result_path = os.path.join(cic.config.DATA_DIR, 'ukwac')
    ukwac = UKWacDataset(ukwac_path, result_save_path=result_path, max_length=max_sentence_length,
                         regenerate=regenerate_dataset)
    print('Len UKWac dataset: %s' % len(ukwac))

    print('Dividing dataset into train/validation split...')
    train_ukwac, val_ukwac = ukwac.split(.9, seed='seed')
    print('Len training set: %s' % len(train_ukwac))
    print('Len validation set: %s' % len(val_ukwac))

    token_to_id, id_to_token = ukwac.get_vocabulary()
    print('Len vocabulary: %s' % len(token_to_id))

    # Create autoencoder
    print('Constructing autoencoder...')
    autoencoder = AutoEncoder(len(token_to_id), tensorboard_name='gmae', save_dir=save_dir,
                              restore_from_save=restore_from_save, max_len=10, rnn_size=rnn_size)

    # Train autoencoder
    if num_epochs > 0:
        print('Training autoencoder...')
        autoencoder.train(train_ukwac, output_tensor_names=['train_prediction'],
                          parameter_dict={'keep prob': 0.9, 'learning rate': learning_rate},
                          num_epochs=num_epochs, batch_size=20, verbose=True)

    # Calculate train accuracy
    print('Calculating training accuracy...')
    results = autoencoder.predict(train_ukwac, output_tensor_names=['train_prediction'])
    np_predictions = results['train_prediction']
    predictions = ukwac.convert_numpy_to_strings(np_predictions)

    total_reconstructions = len(predictions)
    correct_reconstructions = 0

    print('Len predictions: %s' % len(predictions))
    for index in range(len(predictions)):
        each_prediction = predictions[index]
        each_original = ukwac.messages[train_ukwac.indices[index]]
        each_np_prediction = np_predictions[index,:]
        each_np_original = ukwac.np_messages[train_ukwac.indices[index],:]

        if index < 10:
            print('Original: %s' % each_original)
            print('Original numpy: %s' % str(each_np_original))
            print('Reconstruction: %s' % each_prediction)
            print('Reconstruction numpy: %s' % str(each_np_prediction))


        if each_prediction == each_original:
            correct_reconstructions += 1

    print('Training accuracy: %s' % (correct_reconstructions / total_reconstructions))
    print()
    # Calculate validation accuracy
    print('Calculating validation accuracy...')
    results = autoencoder.predict(val_ukwac, output_tensor_names=['train_prediction'])
    np_predictions = results['train_prediction']
    predictions = ukwac.convert_numpy_to_strings(np_predictions)

    total_reconstructions = len(predictions)
    correct_reconstructions = 0

    for index in range(len(predictions)):
        each_prediction = predictions[index]
        each_original = ukwac.messages[val_ukwac.indices[index]]


        if index < 10:
            print('Reconstruction: %s' % each_prediction)
            print('Original: %s' % each_original)

        if each_prediction == each_original:
            correct_reconstructions += 1
    print('Validation accuracy: %s' % (correct_reconstructions / total_reconstructions))

