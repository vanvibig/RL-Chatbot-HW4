# coding=utf-8
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

from collections import defaultdict
import json
import numpy as np
import os
import re
from gensim.models import Word2Vec
from gensim.models import FastText
from pyvi import ViTokenizer
# from gensim.models.wrappers import FastText

class Utils:

    # Constants
    filename_conversations = 'data/movie_conversations.txt'
    filename_lines = 'data/movie_lines.txt'
    filename_word_vec = 'data/word_vec.txt'
    filename_conv_pair_pickle = 'data/conv_pair.p'
    filename_sample_test = 'sample_input.txt'
    filename_test = 'test_input.txt'
    filename_sent_list = 'data/sent_list.txt'

    MARKER_PAD = '<pad>'
    MARKER_BOS = '<s>'
    MARKER_EOS = '</s>'
    MARKER_OOV = '<oov>'
    marker_list = [MARKER_PAD, MARKER_BOS, MARKER_EOS, MARKER_OOV]

    max_seq_len = 20
    min_occurence = 2

    # Output
    raw_conv_pair = None
    conv_pair = None
    sample_test_questions = None
    test_questions = None

    word_freq_dict = None
    word_list = None
    word_index_dict = None
    word_vec_dict = None

    def __init__(self):
        print 'Initializing...'
        self.__extractConvPair()
        self.__genWordList()
        self.__createWordVecDict()
        self.__editWordVecDict()
        self.__genWordVecDict()
        self.__readTestQuestions()

    def __extractConvPair(self):
        # Read movie lines
        movie_lines = dict()
        file = open(self.filename_lines, 'r')
        for line in file:
            line = line[:-1]
            tokens = re.split('\ \+\+\+\$\+\+\+\ ', line)
            
            # pyvi
            # tokens_vi = u' '.join(t for t in tokens)
            # tokens_vi = ViTokenizer.tokenize(tokens_vi)
            # tokens_vi = re.split(' ', tokens_vi)
            # movie_lines[tokens[0]] = tokens_vi
            # end pvi

            movie_lines[tokens[0]] = tokens[4]

        # Read movie conversations
        movie_conv = []
        file = open(self.filename_conversations, 'r')
        for line in file:
            line = line[:-1]
            tokens = re.split('\ \+\+\+\$\+\+\+\ ', line)
            movie_conv.append(json.loads(tokens[3].replace("'", '"')))

        # Make pairs
        raw_conv_pair = []
        for conv in movie_conv:
            for i, sent in enumerate(conv):
                if i == 0:
                    continue
                if movie_lines[conv[i-1]] == '' or movie_lines[conv[i]] == '':
                    continue
                raw_conv_pair.append(
                    [movie_lines[conv[i-1]], movie_lines[conv[i]]])

        # Extract elite conversation pair
        self.conv_pair = conv_pair = []
        for pair in raw_conv_pair:
            q_sents = re.findall('[A-Z][^.!?]*[.!?]', pair[0])
            if q_sents == []:
                continue
            q = self.captionToTokenList(q_sents[-1])
            if len(q) > self.max_seq_len:
                continue

            a_sents = re.findall('[A-Z][^.!?]*[.!?]', pair[1])
            if a_sents == []:
                continue
            a = self.captionToTokenList(a_sents[0])
            if len(a) > self.max_seq_len:
                continue

            conv_pair.append([q, a])
            print q
            print a
        print "run __extractConvPair"
        
    def __genWordList(self):
        self.word_freq_dict = defaultdict(float)
        total_word_count = 0.0
        sent_list_txt = open('data/sent_list.txt', 'w')
        for pair in self.conv_pair:
            for sent in pair:
                # print(u' '.join(s for s in sent if s != sent[-1]))
                # kv
                # for w in self.word_list:
                s = u' '.join(s for s in sent if s != sent[-1])
                sent_list_txt.write(s+'\n')
                # end kv
                for token in sent:
                    if token in self.marker_list:
                        continue
                    self.word_freq_dict[token] += 1
                    total_word_count += 1.0
        temp = defaultdict(float)
        self.word_freq_dict = dict(
            (k, v) for k, v in self.word_freq_dict.iteritems() if v >= self.min_occurence)
        for word in self.word_freq_dict:
            self.word_freq_dict[word] /= total_word_count

        word_freq_list = sorted(
            self.word_freq_dict.iteritems(), key=lambda (k, v): v, reverse=True)
        self.word_list = self.marker_list + [k for (k, v) in word_freq_list]
        
        # kv
        word_list_txt = open('data/word_list.txt', 'w')
        for w in self.word_list:
            word_list_txt.write(w+'\n')
        # end kv

        self.word_index_dict = defaultdict(
            lambda: self.marker_list.index(self.MARKER_OOV))
        self.word_index_dict.update(
            [(self.word_list[i], i) for i in range(len(self.word_list))])

        print "run __genWordList"

    def __createWordVecDict(self):
        train_data = []
        sents = open(self.filename_sent_list, 'r').readlines()
        for sent in sents:
            train_data.append(sent.decode('utf-8').split())
    
        model_fasttext = FastText(size=300, window=5, min_count=2, workers=4, sg=1)
        model_fasttext.build_vocab(train_data)
        model_fasttext.train(train_data, total_examples=model_fasttext.corpus_count, epochs=model_fasttext.iter)
        # model_fasttext.wv.save_word2vec_format("data/fasttext_gensim_word_vec.txt", binary=False)
        model_fasttext.wv.save_word2vec_format("data/word_vec_old.txt", binary=False)
        
        # model = Word2Vec(train_data, size=300, window=5, min_count=2, workers=4, sg=1)
        # model.wv.save_word2vec_format("data/word2vec_skipgram.txt", binary=False)
        
        print "run __createWordVecDict"

    def __editWordVecDict(self):
        new_file_wv = open('data/word_vec.txt', 'w')
        old_file_wv = open('data/word_vec_old.txt', 'r').readlines()
        # remove first line
        old_file_wv = old_file_wv[1:]
        for v in old_file_wv:
            new_file_wv.write(v)
        
        print "run __editWordVecDict"

    def __genWordVecDict(self):
        if not os.path.isfile(self.filename_word_vec):
            return
        self.word_vec_dict = defaultdict(lambda: self.MARKER_OOV)
        with open(self.filename_word_vec, 'r') as file:
            for line in file:
                content = line.split()
                vec = [float(elem) for elem in content[1:]]
                self.word_vec_dict[content[0]] = np.array(
                    vec, dtype=np.float32)
        for marker in self.marker_list:
            self.word_vec_dict[marker] = np.zeros([300], dtype=np.float32)

        print "run __genWordVecDict"

    def __readTestQuestions(self):
        file = open(self.filename_sample_test)
        self.sample_test_questions = []
        for line in file:
            line.replace('\n', '')
            self.sample_test_questions.append(self.captionToTokenList(line))
        file = open(self.filename_test)
        self.test_questions = []
        for line in file:
            line.replace('\n', '')
            self.test_questions.append(self.captionToTokenList(line))

        print "run __readTestQuestions"

    @staticmethod
    def captionToTokenList(caption):
        # Convert to lowercase
        caption = caption.lower()

        # "blabla" -> blabla
        caption = re.sub('"(?P<bla>([a-zA-Z]+))"',
                         lambda m: m.group('bla'), caption)

        # Isolate trailing punctuations
        caption = re.sub('(?P<letter>\w)(?P<punc>\W+)$',
                         lambda m: m.group('letter') + ' ' + m.group('punc'), caption)

        # Isolate comma
        caption = re.sub('(?P<letter>\w),', lambda m: m.group(
            'letter') + ' , ', caption)

        # Isolate 's
        caption = re.sub("(?P<letter>\w)'s", lambda m: m.group(
            'letter') + " 's ", caption)

        # Tokenize
        token_list = re.split('\s+', caption)

        # Add EOS
        token_list.append(Utils.MARKER_EOS)

        return token_list

    @staticmethod
    def tokenListToCaption(token_list):
        # Trim words after EOS
        if Utils.MARKER_EOS in token_list:
            token_list = token_list[:token_list.index(Utils.MARKER_EOS)]
        if token_list == []:
            return ''

        # Convert back to sentence
        caption = reduce(lambda s, term: s + ' ' + term,
                         token_list[1:], token_list[0])

        # Captalize the first letter
        caption = caption[0:1].upper() + caption[1:]

        # Remove space before trailing punctuations
        caption = re.sub(' (?P<punc>\W+)$', lambda m: m.group('punc'), caption)

        # Remove space before comma
        caption = re.sub(' , ', ', ', caption)

        # Remove space before 's
        caption = re.sub(" (?P<punc>'s)", lambda m: m.group('punc'), caption)

        return caption


if __name__ == '__main__':
    utils = Utils()
    # print utils
    # print utils.word_vec_dict
    # file = open('data/word_list.txt', 'w')
    # for word in utils.word_list:
    #     file.write(word + ' ')