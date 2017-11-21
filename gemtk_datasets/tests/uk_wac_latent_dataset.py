"""Test code for the UK Wac latent dataset, where all examples from the UK Wac dataset
are transformed into the latent space of the autoencoder."""
from cic.gemtk_datasets.uk_wac_dataset import UKWacDataset
from cic.gemtk_datasets.uk_wac_latent_dataset import LatentUKWacDataset
from cic.autoencoders.gm_auto_encoder import AutoEncoder
import cic.config
import os

ukwac_path = '/data2/arogers/Corpora/En/UkWac/Plain-txt/ukwac_subset_100M.txt'
result_path = os.path.join(cic.config.DATA_DIR, 'ukwac')
print('Loading dataset...')
ukwac = UKWacDataset(ukwac_path, result_save_path=result_path, max_length=10, regenerate=False)
token_to_id, id_to_token = ukwac.get_vocabulary()
autoencoder = AutoEncoder(len(token_to_id), max_len=10)
latent_ukwac = LatentUKWacDataset(ukwac, autoencoder)
print('Number of latent messages in dataset: %s' % latent_ukwac.np_l_messages.shape[0])
print('Vocabulary size: %s' % len(ukwac.get_vocabulary()[0]))