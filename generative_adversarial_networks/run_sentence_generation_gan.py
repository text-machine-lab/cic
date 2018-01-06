"""Run sentence generation GAN"""

from sacred import Experiment
import os
from cic.generative_adversarial_networks.sentence_generation_gan import SentenceGenerationGAN, GaussianRandomDataset
from cic.gemtk_datasets.latent_uk_wac_dataset import LatentUKWacDataset
from cic.gemtk_datasets.uk_wac_dataset import UKWacDataset
from cic.autoencoders.gm_auto_encoder import AutoEncoder
from arcadian.dataset import MergeDataset, DatasetPtr
import cic.config
import numpy as np

ex = Experiment('sentence_gan')

@ex.config
def config():
    print('Running config')
    code_size = 600
    regenerate_latent_ukwac = True
    max_number_of_sentences = None
    num_generator_layers = 30
    num_discriminator_layers = 30
    sentence_gan_save_dir = os.path.join(cic.config.DATA_DIR, 'sentence_gan')
    num_epochs = 100
    restore_sentence_gan_from_save = False
    gen_learning_rate = 0.0001
    dsc_learning_rate = 0.0001
    max_len = 20
    keep_prob = 1.0

    c = .001  # 10**(-num_discriminator_layers)/code_size  # D weight clipping

@ex.automain
def main(code_size, regenerate_latent_ukwac, max_number_of_sentences, num_generator_layers, num_discriminator_layers,
         sentence_gan_save_dir, num_epochs, restore_sentence_gan_from_save, gen_learning_rate,
         dsc_learning_rate, keep_prob, c, max_len):
    print('Running program')

    # If we will reconstruct latent dataset, construct ukwac dataset and encoder portion of pretrained autoencoder.
    ukwac = None
    encoder = None

    print('Constructing UK Wac dataset')
    ukwac = UKWacDataset(cic.config.UKWAC_PATH, result_save_path=cic.config.UKWAC_RESULT_PATH,
                         max_length=10, regenerate=False, max_number_of_sentences=max_number_of_sentences)
    print('Length of UK Wac dataset: %s' % len(ukwac))

    if regenerate_latent_ukwac:
        print('Regenerating latent UK Wac')
        print('Constructing pre-trained autoencoder')
        encoder = AutoEncoder(len(ukwac.token_to_id), save_dir=cic.config.GM_AE_SAVE_DIR,
                              restore_from_save=True, max_len=max_len, rnn_size=code_size,
                              encoder=True, decoder=False)

    # Build latent ukwac dataset, either by regenerating it or loading from saved results.
    print('Constructing latent UK Wac dataset')
    latent_ukwac = LatentUKWacDataset(os.path.join(cic.config.DATA_DIR, 'latent_ukwac'), code_size,
                                      ukwac=ukwac, autoencoder=encoder, regenerate=regenerate_latent_ukwac)

    z_dataset = GaussianRandomDataset(len(latent_ukwac), code_size, 'z')

    merge_dataset = MergeDataset([latent_ukwac, z_dataset])

    print('Length of Latent UK Wac dataset: %s' % len(latent_ukwac))

    # Construct SentenceGAN.
    gan = SentenceGenerationGAN(code_size=code_size, num_gen_layers=num_generator_layers,
                                num_dsc_layers=num_discriminator_layers,
                                save_dir=sentence_gan_save_dir, tensorboard_name='sentence_gan',
                                restore_from_save=restore_sentence_gan_from_save)

    # Train SentenceGAN.
    if num_epochs > 0:
        gan.train(merge_dataset, parameter_dict={'gen_learning_rate': gen_learning_rate,
                                                 'dsc_learning_rate': dsc_learning_rate,
                                                 'keep_prob': keep_prob,
                                                 'c': c}, num_epochs=num_epochs)

    # Generate and print 100 examples!
    z_examples = GaussianRandomDataset(100, code_size, 'z')
    generated_codes = {'code': gan.predict(z_examples, output_tensor_names=['code'])['code']}

    # generated_codes = DatasetPtr(latent_ukwac, range(100))

    assert len(generated_codes['code']) == 100

    # Create decoder to convert latent sentences back to English
    decoder = AutoEncoder(len(ukwac.token_to_id), save_dir=cic.config.GM_AE_SAVE_DIR,
                          restore_from_save=True, max_len=10, rnn_size=code_size,
                          encoder=False, decoder=True)

    generated_np_sentences = decoder.predict(generated_codes, output_tensor_names=['train_prediction'])['train_prediction']

    assert generated_np_sentences.shape == (len(generated_codes['code']), decoder.max_len)

    generated_sentences = ukwac.convert_numpy_to_strings(generated_np_sentences)

    for each_sentence in generated_sentences:
        print(each_sentence)

