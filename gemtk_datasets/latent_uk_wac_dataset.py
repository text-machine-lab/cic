import numpy as np
import arcadian.dataset
import os
import h5py

class LatentUKWacDataset(arcadian.dataset.Dataset):
    def __init__(self, save_dir, rnn_size, ukwac=None, autoencoder=None, regenerate=False,
                 conversion_batch_size=50):
        """Dataset of UK Wac sentences, encoded using a pretrained autoencoder.
        Sentences are represented as vectors of information, as input to a system
        that requires a compacted, continuous representation of a sentence.

        Arguments:
            - ukwac: instance of UKWac dataset with sentences to convert to latent codes
            - autoencoder: pretrained autoencoder, encoder is used to convert sentences
            - save_dir: path to save results for faster loading
            - regenerate: if True, regenerate codes even if they exist

        ukwac and autencodor must be specified if codes are to be regenerated.
        """
        # if save_dir is not None:
        #     self.saved_codes_filename = os.path.join(save_dir, 'np_ukwac_codes.npy')
        #
        # if save_dir is not None and not regenerate and os.path.isfile(self.saved_codes_filename):
        #     # If we have already saved the codes, load them from save (unless we wish to regenerate them)
        #     self.np_latent_codes = np.load(open(self.saved_codes_filename, 'rb'))
        # else:
        #     assert autoencoder is not None and ukwac is not None
        #     # If codes don't exist, we must generate them.
        #     self.np_latent_codes = autoencoder.predict(ukwac, output_tensors=['code'])
        #     if save_dir is not None:
        #         np.save(self.np_latent_codes, open(self.saved_codes_filename, 'wb'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.latent_code_dataset_filename = os.path.join(save_dir, 'ukwac_codes.hdf5')

        # If the codes haven't been generated, or we wish to regenerate them, convert.
        if regenerate or not os.path.isfile(self.latent_code_dataset_filename):
            self.dataset_file = h5py.File(self.latent_code_dataset_filename, 'w')
            print('Generating latent codes...')
            dataset = self.dataset_file.create_dataset('codes', (len(ukwac), rnn_size), dtype='float32')
            index_in_dataset = 0
            for ukwac_batch in ukwac.generate_batches(batch_size=conversion_batch_size):
                result = autoencoder.predict(ukwac_batch, output_tensors=['code'], batch_size=conversion_batch_size)
                np_batch_codes = result['code']
                dataset[index_in_dataset:index_in_dataset+np_batch_codes.shape[0], :] = np_batch_codes
                index_in_dataset += np_batch_codes.shape[0]
        else:
            print('Codes already exist. Loading from file')
            self.dataset_file = h5py.File(self.latent_code_dataset_filename, 'r+')

        self.dataset = self.dataset_file['codes']

    def __getitem__(self, index):
        # Get a single item as an index from the dataset.
        return {'code': self.dataset[index, :]}

    def __len__(self):
        # Return the length of the dataset.
        return self.dataset.shape[0]

