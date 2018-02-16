from cic.datasets.cornell_movie_conversation import CornellMovieConversationDataset
from cic.datasets.text_dataset import convert_numpy_array_to_strings
import numpy as np

max_s_len = 10

ds = CornellMovieConversationDataset(max_s_len, reverse_inputs=False, seed='seed')

for index in range(len(ds)):
    message, response = ds.examples[index]
    message = ' '.join(message)
    response = ' '.join(response)

    example = ds[index]
    np_message = np.reshape(example['message'], [1, -1])
    np_response = np.reshape(example['response'], [1, -1])

    reconst_message = convert_numpy_array_to_strings(np_message, ds.inverse_vocab,
                                                     stop_token=ds.stop_token, keep_stop_token=True )[0]

    reconst_response = convert_numpy_array_to_strings(np_response, ds.inverse_vocab,
                                                     stop_token=ds.stop_token, keep_stop_token=True)[0]


    if message != reconst_message:
        print('Message: %s' % message)
        print('Reconst: %s' % reconst_message)

    if response != reconst_response:
        print('Response: %s' % response)
        print('Reconstr: %s' % reconst_response)