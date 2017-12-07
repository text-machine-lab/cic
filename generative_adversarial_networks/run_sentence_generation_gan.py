"""Run sentence generation GAN"""

from sacred import Experiment
import os
from cic.generative_adversarial_networks.sentence_generation_gan import SentenceGenerationGAN
from cic.gemtk_datasets.latent_uk_wac_dataset import LatentUKWacDataset
from cic.gemtk_datasets.uk_wac_dataset import UKWacDataset
from cic.autoencoders.gm_auto_encoder import AutoEncoder
import cic.config

ex = Experiment('sentence_gan')

@ex.config
def config():
    print('Running config')
    code_size = 600
    regenerate_latent_ukwac = True

@ex.automain
def main(code_size, regenerate_latent_ukwac):
    print('Running program')

    # If we will reconstruct latent dataset, construct ukwac dataset and pretrained autoencoder.
    ukwac = None
    autoencoder = None
    if regenerate_latent_ukwac:
        ukwac = UKWacDataset(cic.config.UKWAC_PATH, result_save_path=cic.config.UKWAC_RESULT_PATH,
                             max_length=10, regenerate=False, max_number_of_sentences=100)
        autoencoder = AutoEncoder(len(ukwac.token_to_id), save_dir=cic.config.GM_AE_SAVE_DIR,
                                  restore_from_save=True, max_len=10, rnn_size=code_size)

    # Build latent ukwac dataset, either by regenerating it or loading from saved results.
    latent_ukwac = LatentUKWacDataset(os.path.join(cic.config.DATA_DIR, 'latent_ukwac'), code_size,
                                      ukwac=ukwac, autoencoder=autoencoder, regenerate=regenerate_latent_ukwac)

    # Construct SentenceGAN.
    gan = SentenceGenerationGAN(code_size=code_size, trainable=False)

    # Train SentenceGAN.

    # Print generated sentences.


