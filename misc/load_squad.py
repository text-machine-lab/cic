"""Copyright 2017 David Donahue. Loads the SQuAD dataset for experimentation."""
import json
import pprint

from cic import paths

index = 0
with open(paths.SQUAD_TRAIN_SET) as squad_file:
    all = json.load(squad_file)
    data = all['data']
    print("Number of documents: %s" % len(data))
    first_record = data[0]
    print("Keys of each document: %s" % str(first_record.keys()))
    num_paragraphs = len(first_record['paragraphs'])
    print("Number of paragraphs in first document: %s" % num_paragraphs)
    first_paragraph = first_record['paragraphs'][0]
    print("Keys of each paragraph: %s" % str(first_paragraph.keys()))
    num_questions = len(first_paragraph['qas'])
    print("Number of questions in first paragraph: %s" % num_questions)
    for each_qas in first_paragraph['qas']:
        print("Keys of each QAS: %s" % each_qas.keys())
        pprint.pprint(each_qas)
        print()
    context = first_paragraph['context']
    pprint.pprint("Context: %s" % context)
