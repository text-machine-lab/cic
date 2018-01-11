"""Functions and constants related to the Cornell Movie Dialogues corpus for loading
and manipulating conversational data."""
import gensim

DELIMITER = ' +++$+++ '


def construct_examples_from_conversations_and_messages(conversations, id_to_message, max_message_length=None):
    examples = []
    for each_conversation in conversations:
        conversation_message_ids = each_conversation[-1]
        for message_index in range(1, len(conversation_message_ids)):
            first_message_id = conversation_message_ids[message_index - 1]
            second_message_id = conversation_message_ids[message_index]
            if id_to_message[first_message_id] is not None and id_to_message[second_message_id] is not None:
                each_message = id_to_message[first_message_id][-1]
                each_response = id_to_message[second_message_id][-1]
                if max_message_length is None or len(each_message) <= max_message_length and len(each_response) <= max_message_length:
                    examples.append([each_message, each_response])
    return examples


def load_messages_from_cornell_movie_lines(movie_lines_filename, nlp, max_number_of_messages=None,
                                           max_message_length=None, stop_token=None):
    movie_lines_file = open(movie_lines_filename, 'rb')
    messages = []
    line_index = 0
    for message_line in movie_lines_file:
        if max_number_of_messages is None or line_index < max_number_of_messages:
            try:
                message_data = message_line.decode('utf-8').split(DELIMITER)
                message_id = message_data[0]
                character_id = message_data[1]
                movie_id = message_data[2]
                character_name = message_data[3]
                message = message_data[4][:-1]
                tk_message = nlp.tokenizer(message.lower())
                tk_tokens = [str(token) for token in tk_message if str(token) != ' ']
                if stop_token is not None:
                    tk_tokens += [stop_token]
                if max_message_length is None or len(tk_tokens) <= max_message_length:
                    messages.append(tk_tokens)
                    line_index += 1
            except UnicodeDecodeError:
                pass
        else:
            break
    return messages


def load_cornell_movie_dialogues_dataset(movie_conversations_filename, max_conversations_to_load=None):
    """Load movie dialogues corpus and return a list of conversations. Each conversation is between
    two characters, and is represented as a list containing:
    [first_character_id, second_character_id, movie_id, message_ids] where ids are assigned per character
    and per movie. message_ids is a list of ids for each message in the conversation. All ids are strings.

    Returns: list of conversations, and dictionary mapping each message id to the value 'None'"""
    movie_conversations_file = open(movie_conversations_filename, 'r')
    line_index = 0
    conversations = []
    id_to_message = {}
    for each_conversation_line in movie_conversations_file:
        if max_conversations_to_load is None or line_index < max_conversations_to_load:
            conversation_data = each_conversation_line.split(DELIMITER)
            first_character_id = conversation_data[0]
            second_character_id = conversation_data[1]
            movie_id = conversation_data[2]
            message_ids = conversation_data[3][2:-3].split("', '")
            for each_message_id in message_ids:
                id_to_message[each_message_id] = None
            conversations.append([first_character_id, second_character_id, movie_id, message_ids])
        line_index += 1
    return conversations, id_to_message


def load_messages_from_cornell_movie_lines_by_id(id_to_message, movie_lines_filename, stop_token, nlp):
    """Given a mapping from message ids to none (id_to_message[3] == None), find each message id
    in the file movie_lines_filename and store in id_to_message in place of None. Add a stop_token character at the
    end of each message. Pre-process each message by tokenizing with nlp, a spacy tokenizer. Returns nothing."""
    movie_lines_file = open(movie_lines_filename, 'rb')
    for message_line in movie_lines_file:
        try:
            message_data = message_line.decode('utf-8').split(DELIMITER)
            message_id = message_data[0]
            if message_id in id_to_message:
                character_id = message_data[1]
                movie_id = message_data[2]
                character_name = message_data[3]
                message = message_data[4][:-1]
                tk_message = nlp.tokenizer(message.lower())
                tk_tokens = [str(token) for token in tk_message if str(token) != ' '] + [stop_token]
                id_to_message[message_id] = [character_id, movie_id, character_name, message, tk_tokens]
        except UnicodeDecodeError:
            pass


def build_vocabulary_from_messages(id_to_message):
    """Given dictionary mapping from message ids to
    message strings, produce a vocabulary of all words
    and return as mapping from each word to its corresponding
    index in the vocabulary."""
    documents = []
    for key in id_to_message:
        each_message_data = id_to_message[key]
        if each_message_data is not None:
            documents.append(each_message_data[-1])
    dictionary = gensim.corpora.Dictionary([['']], prune_at=None)
    dictionary.add_documents(documents, prune_at=None)
    vocab_dict = dictionary.token2id
    return vocab_dict