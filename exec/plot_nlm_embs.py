"""Plot word embeddings learned by the Neural Language model, using TSNE dimensionality reduction."""
import matplotlib.pyplot
import numpy as np
from sklearn.manifold import TSNE
import cic.paths
import os
import pickle

num_words = 500
embs_path = os.path.join(cic.paths.DATA_DIR, 'nlm_embs.npy')
vocab_path = os.path.join(cic.paths.BOOK_CORPUS_RESULT, 'vocab.pkl')

embs = np.load(open(embs_path, 'rb'))
tk2id = pickle.load(open(vocab_path, 'rb'))

id2tk = {tk2id[k]: k for k in tk2id}

embs = embs[:num_words, :]

# Run TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(embs)

# Plot
print('Plotting...')

fig, ax = matplotlib.pyplot.subplots()
matplotlib.pyplot.scatter(tsne_results[:, 0], tsne_results[:, 1])

matplotlib.pyplot.title('Neural Language Model Word Embeddings Plot')

for id in range(embs.shape[0]):
    ax.annotate(id2tk[id], (tsne_results[id, 0], tsne_results[id, 1]))

matplotlib.pyplot.show()