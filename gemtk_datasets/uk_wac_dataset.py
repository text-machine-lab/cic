"""UK WAC dataset."""
from gemtk_datasets import string_dataset


class UKWacDataset(string_dataset.StringDataset):

    def __init__(self, ukwac_path, result_save_path=None, token_to_id=None,
                 max_length=30, regenerate=False):
        """Instantiate UK Wac dataset from raw ukwac_path file (slow). Store and load
        resulting training examples to and from result_path (fast). Separates dataset
        into training and testing sets and builds vocabulary. Must specify whether to
        load training or test sets, cannot load both!

        Arguments:
            - ukwac_path: absolute or relative location of dataset
            - result_save_path: path to save intermediate results for faster reloading (recommended)
            - token_to_id: provided vocabulary used to convert strings to numpy (regenerates intermediate results)
            - max_length: maximum length of string to be converted to numpy
            - regenerate: ignore intermediate results, regenerate all data
            """
        self.ukwac_path = ukwac_path

        def contains_numbers(s):
            return any(c.isdigit() for c in s)

        # Read all sentences from file, and keep them if they follow the correct formatting.
        results_exist = self._numpy_string_formatting_results_exist(result_save_path)
        #
        filtered_sentences = []
        if not results_exist or regenerate or token_to_id is not None:
            for index, line in enumerate(open(self.ukwac_path, 'r', encoding='utf-8', errors='ignore')):
                sentences = line.split('. ')
                for each_sentence in sentences:
                    each_sentence = each_sentence.lower()
                    #print(each_sentence)
                    if not each_sentence.isspace() \
                            and '(' not in each_sentence \
                            and ')' not in each_sentence \
                            and not each_sentence.startswith('current url') \
                            and not contains_numbers(each_sentence) \
                            and '"' not in each_sentence \
                            and ':' not in each_sentence:
                        each_sentence += '.'
                        each_sentence = each_sentence.strip()
                        if max_length is None or len(each_sentence.split()) < max_length:
                            filtered_sentences.append(each_sentence)

        # Use StringDataset class to automatically tokenize and convert strings to numpy format.
        super().__init__(filtered_sentences,
                         max_length,
                         result_save_path=result_save_path,
                         token_to_id=token_to_id,
                         regenerate=regenerate)