"""Copyright 2017 David Donahue. Loads MS Marco train dataset (without crashing). Explore dataset"""
import json
import pprint
import resource
import string
import sys

from cic import paths

rsrc = resource.RLIMIT_DATA
soft, hard = resource.getrlimit(rsrc)
resource.setrlimit(rsrc, (1024, hard))

count = 0
count_answer_in_passage = 0
total_size_of_all_records = 0
num_examples_to_print = 1
translator = str.maketrans('', '', string.punctuation)
for each_line in open(paths.MS_MARCO_TRAIN_SET):
    record = json.loads(each_line)
    size_of_record = sys.getsizeof(record)
    total_size_of_all_records += size_of_record
    if count < num_examples_to_print:
        if len(record[u'answers']) > 0:
            answer = record[u'answers'][0]
            passages = record[u'passages']
            for each_passage in passages:
                if each_passage['is_selected']:
                    passage_text = each_passage['passage_text']
                    formatted_answer = answer.lower().translate(translator)
                    formatted_text = passage_text.lower().translate(translator)
                    if formatted_answer in formatted_text:
                        count_answer_in_passage += 1
                        break
                    else:
                        pass#print('Answer: %s' % formatted_answer)
                        #print('Passage: %s' % formatted_text)
                        #print()

        #print record[u'answers']
        #print record.keys()
        #print record[u'passages'][0]
        pprint.pprint(record)
        #print()
    count += 1

print('Size of all records: %s' % total_size_of_all_records)
print('Number of records: %s' % count)
print('Percent of answers contained in passages: %s%%' % (count_answer_in_passage / num_examples_to_print * 100))