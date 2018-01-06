"""Train and evaluate neural language model on UK Wac."""

from sacred import Experiment

ex = Experiment('nlm')

@ex.config
def config():
    pass

@ex.main
def main():
    pass