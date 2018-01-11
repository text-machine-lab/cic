from cic.datasets.latent_ae import LatentUKWacDataset
from cic.datasets.uk_wac import UKWacDataset
from cic.ae.gm_autoencoder import AutoEncoder
import os
import cic.config
import numpy as np

import unittest2

# Note: These tests depend on the ukwac dataset being in the correct location on Shala. Depends on
# autoencoder being pretrained with checkpoint in 'ukwac_autoencoder2'. Make sure parameters are correct first.
RNN_SIZE = 600

class LatentUKWacDatasetTest(unittest2.TestCase):
    def setUp(self):
        # Load UK Wac dataset
        print("Loading dataset")
        ukwac_path = '/data2/arogers/Corpora/En/UkWac/Plain-txt/ukwac_subset_100M.txt'
        self.ukwac = UKWacDataset(ukwac_path, result_save_path=cic.config.UKWAC_RESULT_PATH, max_length=10, regenerate=False,
                                  max_number_of_sentences=100)

        # Load autoencoder
        print('Constructing pre-trained autoencoder')
        self.save_dir = cic.config.GM_AE_SAVE_DIR
        print('Autoencoder location: %s' % (self.save_dir))
        self.autoencoder = AutoEncoder(len(self.ukwac.token_to_id), save_dir=self.save_dir,
                                       restore_from_save=True, max_len=10, rnn_size=RNN_SIZE, decoder=False)

    def test_construction_with_regeneration(self):
        latent_ukwac = LatentUKWacDataset(os.path.join(cic.config.DATA_DIR, 'test'), RNN_SIZE, ukwac=self.ukwac,
                                          autoencoder=self.autoencoder, regenerate=True,
                                          conversion_batch_size=50)
        for index in range(len(latent_ukwac)):
            print(np.mean(latent_ukwac[index]['code']))

    def test_construction_without_regeneration(self):
        latent_ukwac = LatentUKWacDataset(os.path.join(cic.config.DATA_DIR, 'test'), RNN_SIZE)
        assert latent_ukwac.dataset is not None
        # assert not np.array_equal(latent_ukwac.dataset, np.zeros(len(latent_ukwac), RNN_SIZE))

    def test_reconstruction(self):
        decoder = AutoEncoder(len(self.ukwac.token_to_id), save_dir=self.save_dir,
                              restore_from_save=True, max_len=10, rnn_size=RNN_SIZE,
                              encoder=False, decoder=True)
        latent_ukwac = LatentUKWacDataset(os.path.join(cic.config.DATA_DIR, 'test'), RNN_SIZE)
        print('Len latent_ukwac: %s' % len(latent_ukwac))
        np_reconstructed_messages = decoder.predict(latent_ukwac, output_tensor_names=['train_prediction'],
                                                             batch_size=20)['train_prediction']
        print('Shape reconstructed messages: %s' % str(np_reconstructed_messages.shape))
        reconstructed_messages = self.ukwac.convert_numpy_to_strings(np_reconstructed_messages)
        print('Number of reconstructed messages: %s' % len(reconstructed_messages))
        assert len(self.ukwac) == len(latent_ukwac)
        # assert len(self.ukwac) == len(reconstructed_messages)
        num_correct_messages = 0
        for index, each_message in enumerate(reconstructed_messages):
            print(each_message)
            original_message = self.ukwac.messages[index]
            print(original_message)
            if each_message == original_message:
                num_correct_messages += 1
        print('Accuracy of reconstruction: %s' % (num_correct_messages / len(self.ukwac)))



if __name__ == '__main__':
    # Allows test to be run both in Pycharm and through command line!
    unittest2.main()