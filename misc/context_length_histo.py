import matplotlib.pyplot as plt
from question_answering import squad_dataset_tools as sdt

from cic import paths

paragraphs = sdt.load_squad_dataset_from_file(paths.SQUAD_TRAIN_SET)
tk_paragraphs = sdt.tokenize_paragraphs(paragraphs)
context_lengths = []
for each_paragraph in tk_paragraphs:
    context = each_paragraph['context']
    context_tokens = context.split()
    context_lengths.append(len(context_tokens))

plt.hist(context_lengths)
plt.title('Context lengths')
plt.xlabel('Length in words')
plt.ylabel('Frequency')
plt.show()