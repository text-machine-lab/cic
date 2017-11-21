"""Mines comments from Reddit as potential sentences for auto-encoder model."""
import pickle
import time

import gensim
import os
import praw
import re

from cic import config

MAX_COMMENT_LENGTH = 20
MAX_RAW_COMMENTS = 100000
MAX_VOCAB_SIZE = 10000
REPEAT_CRAWL_N_TIMES = 33

if not os.path.exists(config.REDDIT_COMMENTS_DUMP):
    os.makedirs(config.REDDIT_COMMENTS_DUMP)

client_id = None
secret = None
agent = None
with open(config.REDDIT_CRAWLER_CREDENTIALS, 'r') as f:
    for line in f:
        line_tokens = line.split()
        if line_tokens[0] == 'client_id':
            client_id = line_tokens[1]
        if line_tokens[0] == 'secret':
            secret = line_tokens[1]
        if line_tokens[0] == 'agent':
            agent = line_tokens[1]

print('client_id: %s' % client_id)
print('secret: %s' % secret)
print('agent: %s' % agent)

reddit = praw.Reddit(client_id=client_id, client_secret=secret, user_agent=agent)

print(reddit.read_only)

subreddit = reddit.subreddit('all')

for j in range(REPEAT_CRAWL_N_TIMES):
    print('Streaming in Reddit comments...')
    raw_comments = []
    for i, comment in enumerate(subreddit.stream.comments()):
        comment_text = comment.body.lower()
        comment_tokens = comment_text.split()
        if len(comment_tokens) <= MAX_COMMENT_LENGTH:
            #print(comment_text)
            raw_comments.append(comment_tokens)
            if len(raw_comments) >= MAX_RAW_COMMENTS:
                break

    dictionary = gensim.corpora.Dictionary(documents=raw_comments)
    dictionary.filter_extremes(no_below=0, no_above=1.0, keep_n=MAX_VOCAB_SIZE)
    vocab_dict = dictionary.token2id
    print('Size of vocabulary: %s' % len(vocab_dict))

    # Remove comments with uncommon vocabulary
    pruned_comments = []
    for each_comment in raw_comments:
        comment_in_vocabulary = True
        each_comment_join = ' '.join(each_comment)
        clean_comment = re.sub('[^A-Za-z0-9\.\,\!\?\']+', ' ', each_comment_join)
        for each_token in clean_comment.split():
            if each_token not in vocab_dict\
                    or 'sub' in each_token\
                    or 'reddit' in each_token\
                    or 'vote' in each_token\
                    or 'link' in each_token\
                    or 'account' in each_token\
                    or '/' in each_token:
                comment_in_vocabulary = False
        if comment_in_vocabulary:
            pruned_comments.append(clean_comment)

    num_examples_print = 10
    for each_comment in pruned_comments[:num_examples_print]:
        print(each_comment)

    print()
    print('Number of raw comments: %s' % len(raw_comments))
    print('Number of pruned comments: %s' % len(pruned_comments))

    moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
    new_filename = os.path.join(config.REDDIT_COMMENTS_DUMP, 'reddit_comments' + moment + '.pkl')
    f = open(new_filename, 'wb')

    print('\nSaving...')
    pickle.dump(pruned_comments, f)
    print('Done')



