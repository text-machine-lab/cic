"""In this script, we plot sentences from the Toronto Book Corpus dataset, and sentences generated
from our fully-trained sentence GAN. We use TSNE for dimensionality reduction."""
import cic.config
from cic.datasets.latent_ae import LatentDataset
from cic.models.sgan import SentenceGenerationGAN, GaussianRandomDataset
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot
import random
import numpy as np

num_plot = 1000

code_size = 100
num_gen_layers = 40
num_dsc_layers = 40
num_dsc_trains = 10
sentence_gan_save_dir = os.path.join(cic.config.DATA_DIR, 'sentence_gan')
max_len = 20

ds = LatentDataset(os.path.join(cic.config.DATA_DIR, 'latent_ukwac'), code_size,
                   data=None, autoencoder=None, regenerate=False)

gan = SentenceGenerationGAN(code_size=code_size, num_gen_layers=num_gen_layers,
                            num_dsc_layers=num_dsc_layers,
                            save_dir=sentence_gan_save_dir, tensorboard_name='sentence_gan',
                            restore=True, num_dsc_trains=num_dsc_trains)

# Gather real examples
random_indices = [int(random.random() * len(ds)) for i in range(num_plot)]

real_s = []
for index in random_indices:
    real_s.append(ds[index]['code'])
real_data = np.stack(real_s, axis=0)

print('Real data shape: %s' % str(real_data.shape))

# Gather fake examples
z = GaussianRandomDataset(num_plot, code_size, 'z')
fake_data = gan.predict(z, outputs=['code'])

print('Fake data shape: %s' % str(fake_data.shape))

all_data = np.concatenate([real_data, fake_data], axis=0)

# Run TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(all_data)

colors = ['b'] * num_plot + ['r'] * num_plot

# Plot
print('Plotting...')

matplotlib.pyplot.title('Real v.s. Fake Latent Space Embeddings')
matplotlib.pyplot.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors)
matplotlib.pyplot.show()