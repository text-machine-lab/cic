"""Train the Generic Model autoencoder on the UKWac dataset. Evaluate its
accuracy."""
import os

from cic import config as config
from cic.autoencoders.gm_auto_encoder import AutoEncoder
from cic.gemtk_datasets.uk_wac_dataset import UKWacDataset

MAX_SENTENCE_LENGTH = 10
SAVE_DIR = os.path.join(config.DATA_DIR, 'ukwac_autoencoder')
RESTORE_FROM_SAVE = False
NUM_EPOCHS = 1000
REGENERATE_DATASET = False  # If problem with dataset, try this first

# Load UKWac dataset
print('Loading dataset...')
ukwac_path = '/data2/arogers/Corpora/En/UkWac/Plain-txt/ukwac_subset_100M.txt'
result_path = os.path.join(config.DATA_DIR, 'ukwac')
ukwac = UKWacDataset(ukwac_path, result_save_path=result_path, max_length=MAX_SENTENCE_LENGTH,
                     regenerate=REGENERATE_DATASET)
print('Len UKWac dataset: %s' % len(ukwac))

print('Dividing dataset into train/validation split...')
train_ukwac, val_ukwac = ukwac.split(.9, seed='seed')
print('Len training set: %s' % len(train_ukwac))
print('Len validation set: %s' % len(val_ukwac))

token_to_id, id_to_token = ukwac.get_vocabulary()
print('Len vocabulary: %s' % len(token_to_id))

# Create autoencoder
print('Constructing autoencoder...')
autoencoder = AutoEncoder(len(token_to_id), tensorboard_name='gmae', save_dir=SAVE_DIR,
                          restore_from_save=RESTORE_FROM_SAVE, max_len=10, rnn_size=500)

# Train autoencoder
if NUM_EPOCHS > 0:
    print('Training autoencoder...')
    autoencoder.train(train_ukwac, output_tensor_names=['train_prediction'],
                      parameter_dict={'keep prob': 0.9, 'learning rate': .001},
                      num_epochs=NUM_EPOCHS, batch_size=20, verbose=True)

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

