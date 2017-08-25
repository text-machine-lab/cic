# cic
Text Machine: Scripts and models for the Conversational Intelligence Challenge are stored here. 

Directory contains the following models:

- Variational autoencoder model with teacher forcing (auto_encoder_model.py)
- Sequence-to-Sequence message-to-response chat model (chat_model.py)
- Autoencoder latent message-to-response chat model (latent_chat_model.py)
- Question-Answering model based on "Machine Comprehension with Match-LSTM and Answer Pointer" (baseline_model.py)

For each model, the \_func.py file contains supporting functions and objects, while the \_model.py processes data, trains the model and makes predictions.

Must add the following to the data directory:
- the Cornell Movie Dialogues dataset directory
- the SQuAD dataset directory

*Reddit data not available publicly.*

Other files contain supporting dataset processing and parsing functions. Documentation is sparse for the moment.

CHECK OUT RECENTLY UPDATED FEATURE BRANCHES, MAIN DOES NOT CONTAIN LATEST FEATURES.
