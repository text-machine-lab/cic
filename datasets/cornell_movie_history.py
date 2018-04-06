"""Cornell Movie Dialogues corpus, where conversations are represented each as a (history, response) pair,
where history are the N previous messages and response is how the systems responds."""
import cic.utils.mdd_tools as mddt
import cic.paths as paths
import spacy
import numpy as np

N = 3
stop_token = '<STOP>'
nlp = spacy.load('en')
num_convos = None # load all convos

conversations, id_to_message = mddt.load_cornell_movie_dialogues_dataset(paths.CORNELL_MOVIE_CONVERSATIONS_FILE,
                                                                         max_conversations_to_load=num_convos)

print('Number of valid conversations: %s' % len(conversations))

convo_lens = [len(conversation[3]) for conversation in conversations]
print('Avg convo len: %s' % np.mean(convo_lens))

print('Finding messages...')
mddt.load_messages_from_cornell_movie_lines_by_id(id_to_message, paths.CORNELL_MOVIE_LINES_FILE, stop_token, nlp)

print(conversations[0])
for index, key in enumerate(id_to_message):
    print(key)
    print(id_to_message[key])

    if index > 10:
        break

examples = []

for each_convo in conversations:
    msg_ids = each_convo[3]
    msg_infos = [id_to_message[msg_id] for msg_id in msg_ids]
