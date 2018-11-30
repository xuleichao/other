'''
by xlc time:2018-08-28 08:27:06
'''
import sys
#sys.path.append('D:/svn_codes/source/public_fun')
import os
main_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
main_path = '/'.join(main_path.split('/')[:-1])
import pandas as pd
import numpy as np
import jieba, copy
from sklearn.feature_extraction.text import CountVectorizer

class CorpusUtils:
    """
    ## 语料数据处理
    1. 处理语料处理为one_hot
    2. 字符到数字的字典
    3. 数字到字符的字典
    """

    def __init__(self, corpus_data):
        self.raw_data = corpus_data # 原始数据
        self.words_num = len(self.raw_data)
        self.add_special()
        self.vocabulary_reverse()
        self.gnrt_onehot_dct()
        
        
    def add_special(self):
        # add `bos` `pad` `eos` `unk`
        self.raw_data['bos'] = self.words_num + 0
        self.raw_data['pad'] = self.words_num + 1
        self.raw_data['eos'] = self.words_num + 2
        self.raw_data['unk'] = self.words_num + 3
    
    def corpus_one_hot(self, a_number):
        
        zero_matrix = np.zeros((self.words_num + 4))
        zero_matrix[a_number] = 1
        return zero_matrix

    def gnrt_onehot_dct(self):
        onehot_dct = {}
        for k, v in self.raw_data.items():
            onehot_dct[k] = self.corpus_one_hot(v)
        self.onehot_dct = onehot_dct
        return onehot_dct

    def vocabulary_reverse(self):
        reverse_dct = {v: k for k, v in self.raw_data.items()}
        self.reverse_dct = reverse_dct
    
    def return_data(self):
        return self.raw_data, self.reverse_dct

def data_utils(corpus_num=100):

    def _del_space(lst):
        # 删除英文空格
        lst = [[j for j in i if j!=' '] for i in lst]
        return lst
        
    data_path = './cmn-eng/cmn.txt'
    data = pd.read_csv(data_path, sep='\t', names=['eng', 'chi'])[: corpus_num]
    eng_tokenize_lst = []
    chi_tokenize_lst = []
    for i in range(corpus_num):
        eng_data = data.iloc[i]['eng']
        chi_data = data.iloc[i]['chi']
        eng_token_lst = jieba.lcut(eng_data)
        chi_token_lst = jieba.lcut(chi_data)
        eng_tokenize_lst.append(eng_token_lst)
        chi_tokenize_lst.append(chi_token_lst)
    #eng_tokenize_lst = _del_space(eng_tokenize_lst)
    return eng_tokenize_lst, chi_tokenize_lst
        
def get_data(language='eng', num=100):
    data_path = './cmn-eng/cmn.txt'
    data = pd.read_csv(data_path, sep='\t', names=['eng', 'chi'])[:num]
    tokenize_lst = []
    for i in range(data.shape[0]):
        eng_data = data.iloc[i][language]
        #chi_data = data.iloc[i]['chi']
        eng_token_lst = jieba.lcut(eng_data)
        #chi_token_lst = jieba.lcut(chi_data)
        tokenize_lst += eng_token_lst
        #tokenize_lst += chi_token_lst
    #counter = CountVectorizer()
    #counter.fit_transform(tokenize_lst)
    #vocabulary = counter.vocabulary_
    #words_num = len(vocabulary)
    #print(vocabulary)
    words_set = list(set(tokenize_lst))
    print(len(words_set))
    vocabulary = {}
    for i in range(len(words_set)):
        vocabulary[words_set[i]] = i
    mycls = CorpusUtils(vocabulary)
    rdata = mycls.return_data()
    return rdata

def sents2onehot(sents_lst, dictionary):
    new_sents_lst = []
    for i in sents_lst:
        vec = dictionary.get(i)
        if vec is None:
            vec = dictionary.get('unk')
        new_sents_lst.append(vec)
    return new_sents_lst

def gnrt_learn_data(language='eng', num_of_data=100):
    X, y = data_utils(num_of_data) #
    str2onehot_dict, _ = get_data(language, num_of_data)
    str2onehot_dict_chi, _ = get_data('chi', num_of_data)
    #print(X)
    #print(str2onehot_dict)
    X = [sents2onehot(i, str2onehot_dict) for i in X]
    y = [sents2onehot(i, str2onehot_dict_chi) for i in y]
    return X, y, str2onehot_dict, str2onehot_dict_chi
    
def data_padding(inputs_lst, max_padding, dictionary):
    new_inputs_lst = []
    for i in inputs_lst:
        if len(i) >= max_padding:
            new_inputs_lst.append(i[: max_padding])
        else:
            for j in range(abs(len(i)-max_padding)):
                i += [dictionary['pad']]
            new_inputs_lst.append(i)
    return new_inputs_lst

def target_utils(target, max_padding, dictionary, is_target=False):
    new_target = []
    if is_target == False:
        for i in target:
            i.insert(0, dictionary['bos'])
    if is_target == True:
        for i in target:
            i.append(dictionary['eos'])
    for i in target:
        if len(i) >= max_padding:
            new_target.append(i[: max_padding])
        else:
            for j in range(abs(len(i)-max_padding)):
                i += [dictionary['pad']]
            new_target.append(i)
    #if is_target == False:
    #    for i in target:
    #        i.append(dictionary['eos'])
    if is_target == True:
        decoder_target = [i[: -1] for i in new_target]
        return decoder_target
    else:
        decoder_input = [i[: -1] for i in new_target]
        return decoder_input
    return decoder_target, decoder_input

def batch_data(data_lst, batch_size):
    # 将 dataflow_loads 的返回值变为batch
    # 数据量的大小
    max_length = len(data_lst[0])
    batch_count = max_length // batch_size
    for i in range(batch_count):
        start = i * batch_size
        end = start + batch_size
        if end >=  max_length:
            end = max_length
        batch_data1 = [data_lst[0][start: end], data_lst[1][start: end],\
                      data_lst[2][start: end]]
        yield batch_data1

def dataflow_loads(language='eng', num_of_data=100):
    X, y, dct, dct_chi = gnrt_learn_data(language, num_of_data)
    y1 =copy.deepcopy(y)
    X = data_padding(X, 15, dct)
    decoder_target = target_utils(y, 20, dct_chi, True)
    decoder_input = target_utils(y1, 20, dct_chi, False)
    return X, decoder_target, decoder_input
    
if __name__ == '__main__':
    X, decoder_target, decoder_input = dataflow_loads()
