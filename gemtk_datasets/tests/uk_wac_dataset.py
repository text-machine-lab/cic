""" Example driver program for constructing a UKWacDataset object. """
from gemtk_datasets.uk_wac_dataset import UKWacDataset
import config
import os


ukwac_path = '/data2/arogers/Corpora/En/UkWac/Plain-txt/ukwac_subset_100M.txt'
result_path = os.path.join(config.DATA_DIR, 'ukwac')
print('Result path: %s' % result_path)
print('config.DATA_DIR: %s' % config.DATA_DIR)
print('Loading dataset...')
ukwac = UKWacDataset(ukwac_path, result_save_path=result_path, max_length=20)
print('Number of numpy messages in dataset: %s' % ukwac.np_messages.shape[0])

for index in range(len(ukwac)):
    if index < 100:
        print(ukwac.formatted_and_filtered_strings[index])
        print(ukwac.np_messages[index, :])
        print(ukwac[index])