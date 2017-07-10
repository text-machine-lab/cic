"""Holds conversations by processing messages and produces responses within the latent space created
by the auto-encoder trained in auto_encoder_model.py"""
import auto_encoder_func as aef
import squad_dataset_tools as sdt
import tensorflow as tf
import pickle
import config
import numpy as np
import spacy
import chat_model_func
import baseline_model_func

LEARNING_RATE = .0001
NUM_EXAMPLES = None
BATCH_SIZE = 20
NUM_EPOCHS = 100
VALIDATE_INPUTS = False
NUM_LAYERS = 60

vocab_dict = pickle.load(open(config.AUTO_ENCODER_VOCAB_DICT, 'rb'))

with tf.Graph().as_default() as encoder_graph:
    encoder = aef.AutoEncoder(aef.LEARNED_EMBEDDING_SIZE, len(vocab_dict),
                                            aef.RNN_HIDDEN_DIM,
                                            aef.MAX_MESSAGE_LENGTH, encoder=True, decoder=False,
                                            save_dir=config.AUTO_ENCODER_MODEL_SAVE_DIR,
                                            load_from_save=True,
                                            learning_rate=aef.LEARNING_RATE,
                                            variational=False)

with tf.Graph().as_default() as decoder_graph:
    decoder = aef.AutoEncoder(aef.LEARNED_EMBEDDING_SIZE, len(vocab_dict),
                                            aef.RNN_HIDDEN_DIM,
                                            aef.MAX_MESSAGE_LENGTH, encoder=False, decoder=True,
                                            save_dir=config.AUTO_ENCODER_MODEL_SAVE_DIR,
                                            load_from_save=True,
                                            learning_rate=aef.LEARNING_RATE,
                                            variational=False)

nlp = spacy.load('en')

examples, np_message, np_response, vocab_dict, vocabulary = chat_model_func.preprocess_all_cornell_conversations(nlp, vocab_dict=vocab_dict,
                                                                                                                 reverse_inputs=False)

if NUM_EXAMPLES is not None:
    examples = examples[:NUM_EXAMPLES]
    np_message = np_message[:NUM_EXAMPLES, :]
    np_response = np_response[:NUM_EXAMPLES, :]

non_duplicate_messages = []
num_duplicates = 0
for example in examples:
    each_message = example[0]
    if each_message not in non_duplicate_messages:
        non_duplicate_messages.append(each_message)
    else:
        num_duplicates += 1
print('Number of duplicates: %s' % num_duplicates)

if NUM_EXAMPLES is None or NUM_EXAMPLES > 10:
    for i in range(10):
        print(examples[i])
else:
    for example in examples:
        print(example)

num_examples = len(examples)

print('Converting messages into latent space')
np_latent_message = encoder.encode(np_message, aef.BATCH_SIZE)
np_latent_response = encoder.encode(np_response, aef.BATCH_SIZE)
print(np_latent_message.shape)
print(np_latent_response.shape)

if VALIDATE_INPUTS:
    print('Reconstructing messages from latent space')
    np_message_reconstruct = decoder.decode(np_latent_message, aef.BATCH_SIZE)
    np_response_reconstruct = decoder.decode(np_latent_response, aef.BATCH_SIZE)
    print(np_message_reconstruct.shape)
    print(np_response_reconstruct.shape)

    message_reconstruct = sdt.convert_numpy_array_to_strings(np_message_reconstruct, vocabulary,
                                                                      stop_token=aef.STOP_TOKEN,
                                                                      keep_stop_token=True)
    response_reconstruct = sdt.convert_numpy_array_to_strings(np_response_reconstruct, vocabulary,
                                                                      stop_token=aef.STOP_TOKEN,
                                                                      keep_stop_token=True)

    print('Shape np_latent_message: %s' % str(np_latent_message.shape))
    print('Shape np_latent_response: %s' % str(np_latent_response.shape))
    print(len(message_reconstruct))
    print(len(response_reconstruct))
    assert len(message_reconstruct) == len(response_reconstruct)
    num_messages_correctly_reconstructed = 0
    num_responses_correctly_reconstructed = 0
    for i in range(len(message_reconstruct)):
        original_message = ' '.join(examples[i][0])
        original_response = ' '.join(examples[i][1])
        if message_reconstruct[i] == original_message:
            num_messages_correctly_reconstructed += 1
        if response_reconstruct[i] == original_response:
            num_responses_correctly_reconstructed += 1
        if i < 5:
            print('Message: ' + message_reconstruct[i] + '|||' + original_message)
            print('Response: ' + response_reconstruct[i] + '|||' + original_response)

    print('Fraction of correctly encoded messages: %s' % (num_messages_correctly_reconstructed / num_examples))
    print('Fraction of correctly encoded responses: %s' % (num_responses_correctly_reconstructed / num_examples))

print(np.mean(np_latent_message))
print(np.std(np_latent_message))
print(np.max(np_latent_message))

np_mean_latent_response = np.mean(np_latent_response, axis=0)
mean_latent_loss = np.mean(np.square(np_latent_response - np_mean_latent_response))
print('Baseline mean latent loss (no access to input): %s' % mean_latent_loss)

tf_latent_message = tf.placeholder(tf.float32, shape=(None, aef.RNN_HIDDEN_DIM), name='latent_message')
tf_latent_response = tf.placeholder(tf.float32, shape=(None, aef.RNN_HIDDEN_DIM), name='latent_response')

# too many layers
# tf_layer1, w1, _ = baseline_model_func.create_dense_layer(tf_latent_message, aef.RNN_HIDDEN_DIM, aef.RNN_HIDDEN_DIM * 2, activation='relu')
# tf_layer2, w2, _ = baseline_model_func.create_dense_layer(tf_layer1, aef.RNN_HIDDEN_DIM * 2, aef.RNN_HIDDEN_DIM * 2, activation='relu')
# tf_layer3, w3, _ = baseline_model_func.create_dense_layer(tf_layer2, aef.RNN_HIDDEN_DIM * 2, aef.RNN_HIDDEN_DIM * 2, activation='relu')
# tf_layer4, w4, _ = baseline_model_func.create_dense_layer(tf_layer3, aef.RNN_HIDDEN_DIM * 2, aef.RNN_HIDDEN_DIM * 2, activation='relu')
# tf_latent_prediction, w5, _ = baseline_model_func.create_dense_layer(tf_layer2, aef.RNN_HIDDEN_DIM * 2, aef.RNN_HIDDEN_DIM, activation=None)

tf_input = tf_latent_message
for i in range(NUM_LAYERS):
    tf_relu, _, _ = baseline_model_func.create_dense_layer(tf_input, aef.RNN_HIDDEN_DIM, aef.RNN_HIDDEN_DIM, activation='relu', std=.001)
    tf_output, _, _ = baseline_model_func.create_dense_layer(tf_relu, aef.RNN_HIDDEN_DIM, aef.RNN_HIDDEN_DIM, activation=None, std=.001)
    tf_input = tf_input + tf_output

tf_latent_prediction = tf_input

tf_total_loss = tf.reduce_mean(tf.square(tf_latent_response - tf_latent_prediction))

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf_total_loss)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for epoch in range(NUM_EPOCHS):
    print('Epoch: %s' % epoch)
    count = 0
    per_print_losses = []
    latent_batch_gen = chat_model_func.BatchGenerator([np_latent_message, np_latent_response], BATCH_SIZE)

    for np_message_batch, np_response_batch in latent_batch_gen.generate_batches():
        assert np_message_batch.shape[0] != 0
        np_batch_loss, np_batch_response, _ = sess.run([tf_total_loss, tf_latent_prediction, train_op], feed_dict={tf_latent_message: np_message_batch,
                                                                    tf_latent_response: np_response_batch})
        per_print_losses.append(np_batch_loss)

        count += 1

    print('Message std: %s' % np.std(np_message_batch))
    print('Response std: %s' % np.std(np_response_batch))
    print('Prediction std: %s' % np.std(np_batch_response))
    print('Loss: %s' % np.mean(per_print_losses))
    #print('Layers: %s, %s, %s, %s, %s' % (np.std(w1.eval()), np.std(w2.eval()), np.std(w3.eval()), np.std(w4.eval()), np.std(w5.eval())))

print('Speak to the model')
while True:
    your_message = input('Message: ')
    if your_message == 'exit':
        break
    np_your_message = aef.convert_string_to_numpy(your_message, nlp, vocab_dict)
    np_your_latent_message = encoder.encode(np_your_message, aef.BATCH_SIZE)
    np_model_latent_response = sess.run(tf_latent_prediction, feed_dict={tf_latent_message: np_your_latent_message})
    np_model_response = decoder.decode(np_model_latent_response, aef.BATCH_SIZE)
    model_response = sdt.convert_numpy_array_to_strings(np_model_response, vocabulary,
                                                                  stop_token=aef.STOP_TOKEN,
                                                                  keep_stop_token=True)[0]
    print('Response: %s' % model_response)

