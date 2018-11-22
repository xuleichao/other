'''

by xlc time:2018-11-22 09:28:13
'''
import sys
#sys.path.append('D:/svn_codes/source/public_fun')
import os
main_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
main_path = '/'.join(main_path.split('/')[:-1])
import numpy as np
import jieba, json
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

class CorpusUtil:
    
    def __init__(self):
        self.corpus_read()

    def corpus_read(self):
        # 读取语料库
        f = open('./cmn-eng/cmn.txt', 'r', encoding='utf-8').readlines()[:10]
        self.raw_corpus = [i.strip().split('\t') for i in f]

    def jieba_utils(self):
        self.eng = [jieba.lcut(i[0]) for i in self.raw_corpus]
        self.chi = [jieba.lcut(i[1]) for i in self.raw_corpus]
        self.eng_split = [' '.join(i) for i in self.eng]

    def feature_extract(self):
        vector = CountVectorizer()
        matrix = vector.fit_transform(self.eng_split)
        return matrix
    
    def json_write(self, data, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False))
        
    def word_num_transform(self, words_lst, name=''):
        # 将词语与数字字典相互转化
        all_words = [j for i in words_lst for j in i]
        all_words_set = list(set(all_words))
        self.word2int_dct = {}
        for i in range(len(all_words_set)):
            self.word2int_dct[all_words_set[i]] = i
        self.word2int_dct['UNK'] = len(all_words_set)
        self.word2int_dct['PAD'] = len(all_words_set) + 1
        self.word2int_dct['EOS'] = len(all_words_set) + 2
        self.word2int_dct['BOS'] = len(all_words_set) + 3
        self.int2word_dct = {j:i for i, j in self.word2int_dct.items()}
        # 字典写入文件中
        self.json_write(self.word2int_dct, './word2int_dct_%s.json'%name)
        self.json_write(self.int2word_dct, './int2word_dct_%s.json'%name)
    
    def onehot_util(self, data_lst, max_length):
        new_lst = []
        for i in data_lst:
            i = [self.word2int_dct[j] for j in i]
            new_lst.append(i)
        # 转变为onehot
        new_lst = self.add_padding(new_lst, max_length)
        result = []
        for i in new_lst:
            sub = []
            for j in range(len(i)):
                zeros = np.zeros(len(self.word2int_dct))
                zeros[i[j]] = 1
                sub.append(zeros)
            result.append(sub)
        return result

    def add_padding(self, data_lst, max_length):
        #chi:10
        #eng:15
        new_lst = []
        for i in data_lst:
            if len(i) >= max_length:
                new_lst.append(i[:max_length])
            else:
                for j in range(abs(len(i)-max_length)):
                    i.append(self.word2int_dct['PAD'])
                    i.insert(0, self.word2int_dct['BOS'])
                    i.append(self.word2int_dct['EOS'])
                new_lst.append(i)
        return new_lst
    
if __name__ == '__main__':
    A = CorpusUtil()
    A.jieba_utils()
    A.word_num_transform(A.eng, 'eng')
    a = A.onehot_util(A.eng, 15)
    
