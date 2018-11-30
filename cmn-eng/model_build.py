'''
by xlc time:2018-08-23 09:50:18
'''
import sys
#sys.path.append('D:/svn_codes/source/public_fun')
import os
main_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
main_path = '/'.join(main_path.split('/')[:-1])
from CorpusUtils import get_data, dataflow_loads
from CorpusUtils import batch_data
import tensorflow as tf
from tensorflow.python.layers.core import Dense # 全连接类
import numpy as np
import pandas as pd
dictionary_eng = get_data()
dictionary_chi = get_data('chi')
import json

def json_write(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))

# 写入字典
json_write(dictionary_eng, 'dictionary_eng')
json_write(dictionary_chi, 'dictionary_chi')

    
def build_embedding(units_num, embedding_length, name='embedding'):
    # embedding 层
    embedding = tf.get_variable(name, [units_num, embedding_length])
    return embedding

def build_lstm(lstm_size, keep_prob, layers_num, name='lstm'):
    # lstm层 build
    def get_a_lstm(lstm_size, keep_prob):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    with tf.name_scope(name):
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_lstm(lstm_size, keep_prob) for i in range(layers_num)])
    return cell

def build_input(input_index, embedding_size=None, name='de'):
    # 输入层 build
    """
    - embeddinng_size: ListType Length>=2
    """
    embedding_layer = build_embedding(embedding_size[0], embedding_size[1])
    embedding_out = tf.nn.embedding_lookup(embedding_layer, input_index, name=name)
    return embedding_out
        
def build_encoder():
    # 编码层 build
    
    pass

def build_decoder():
    # 解吗层 build
    pass

def build_inference():
    # 推理层 build
    pass

def corpus_util(corpus_data):
    # 语料处理 one_hot
    pass

if __name__ == '__main__':
    data = pd.read_csv('./cmn-eng/cmn.txt', sep='\t', names=['eng', 'chi'])
    #最大英文单词数 32
    #最大中文字符数 44
    lstm_hidden_num = 1000 # lstm 隐层维度
    words_class = 150 # 单词的种类数
    input_max_length = 15 # 最大输入长度
    decoder_input_max_length = 19
    batch_size = 10 #
    fully_connect = 130
    
    # 建立input
    inputs = tf.placeholder(tf.int32, shape=(None, input_max_length), name='inputs')
    decoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name='decoder_inputs')
    targets = tf.placeholder(tf.int32, shape=(None, decoder_input_max_length), name='targets')
    # 词嵌入层
    embedding_layer = build_embedding(words_class, 130)
    embedding_layer_chi = build_embedding(150, 130, name='chi')# 解码词语嵌入
    # lstm 输入
    lstm_inputs = tf.nn.embedding_lookup(embedding_layer, inputs)
    # 编码层
    encoder_layer = build_lstm(lstm_hidden_num, 0.9, 1)
    # 初始化lstm 层
    initial_state = encoder_layer.zero_state(batch_size, tf.float32)
    # lstm 动态时间维度展开
    encoder_lstm_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_layer, lstm_inputs, dtype=tf.float32)
    # 开始构建解码层
    # 解码层第一个传入的字符为`<BOE>`
    encoder_final_c = encoder_final_state[0][0] # 编码层最后的状态 c
    # 全连接层
    F_C_layer = Dense(fully_connect)
    #decoder_lstm_inputs = tf.nn.embedding_lookup(embedding_layer_chi, decoder_inputs)
    #decoder_layer = build_lstm(lstm_hidden_num, 0.9, 1)
    #decoder_layer = tf.contrib.rnn.LSTMCell(100)
    #decoder_lstm_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_layer, decoder_lstm_inputs, initial_state=\
    #                                                              encoder_final_state, scope="plain_decoder")# 坑啊，两个dynamic 要用scope进行区分

    #decoder_lstm_outputs, decoder_final_state = inference_layer(decoder_inputs, encoder_final_state, is_inference=False)
    # 构建损失函数
    
    #decoder_logits = tf.to_float(decoder_logits, name='ToFloat32')
    
    # 解码层构建-for inference
    def inference_layer(inputs_infer, initial_state=None, is_inference=True):
        """
        ## 刚发现编码层和推理层的关系，所以决定修改这块，改成编码层和推理层公用
        - 编码层直接可以进行编码，经过一次dynamic_rnn 就可以
        - 如果是推理层，就需要进行将推理结果拼接起来返回，所以这就是concat 的作用吗？
        推理层构建
        - 结束条件
          1. 达到最大字符数
          2. 输出<END>
        """
        global dictionary_eng, dictionary_chi
        global embeddinng_layer, embedding_layer_chi
        global encoder_layer
        global decoder_layer
        global lstm_hidden_num
        #global encoder_final_state
        if is_inference == False:
            decoder_lstm_inputs = tf.nn.embedding_lookup(embedding_layer_chi, inputs_infer)
            decoder_layer = build_lstm(lstm_hidden_num, 0.9, 1)
            #decoder_layer = tf.contrib.rnn.LSTMCell(100)
            decoder_lstm_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_layer, decoder_lstm_inputs, initial_state=\
                                                                          initial_state, scope="plain_decoder")# 坑啊，两个dynamic 要用scope进行区分
            return decoder_lstm_outputs, decoder_final_state
        stop_condition = False
        count = 0
        # 编码
        inputs_encode = tf.nn.embedding_lookup(embedding_layer, inputs_infer)
        encoder_lstm_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_layer, inputs_encode, dtype=tf.float32)
        # 推断初始输入
        inference_initial_inputs = tf.nn.embedding_lookup(embedding_layer_chi, [[dictionary_chi[0]['bos']]])
        string = ''
        while not stop_condition:
            # 预测字符
            #判断是否终止
            #需要向量转字符
            #print(11)
            inference_outputs, infenrence_state = tf.nn.dynamic_rnn(decoder_layer, inference_initial_inputs, \
                                                                    initial_state=encoder_final_state,scope="plain_decoder")
            inference_outputs = F_C_layer(inference_outputs)
            
            a, b = sess.run([inference_outputs, tf.argmax(inference_outputs, 2)])
            outputs_str = dictionary_chi[1].get(b[0][0], '<unk>')
            string += outputs_str
            print(outputs_str, count)
            if outputs_str == 'eos' or count > 20:
                stop_condition = True
            else:
                inference_initial_inputs = inference_outputs
                encoder_final_state = infenrence_state
            count += 1
            print(string)
        return string

    def test_model(inputs):
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint('model/')
        saver.restore(sess, model_file)
        inference_layer(inputs)
    
    decoder_lstm_outputs, decoder_final_state = inference_layer(decoder_inputs, encoder_final_state, is_inference=False)
    #infer_outputs, _ = inference_layer(decoder_inputs, None, is_inference=True)
    
    #decoder_logits = tf.contrib.layers.linear(decoder_lstm_outputs, fully_connect)
    decoder_logits = F_C_layer(decoder_lstm_outputs)
    #infer_logits = F_C_layer(infer_outputs)
    decoder_logits_argmax = tf.argmax(decoder_logits, 2)
    labels_target = tf.one_hot(targets, depth=fully_connect, dtype=tf.float32)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels_target,
        logits=decoder_logits
    )
    loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # begin train
    # 训练参数定义
    epochs = 1000 # 训练次数
    encoder_inputs, decoder_target, decoder_input = dataflow_loads(num_of_data=100)
    # 生成batch 数据
    batch_size = 100
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    count = 0
    for ep in range(epochs):
        data_with_batch = batch_data([encoder_inputs, decoder_target, decoder_input], batch_size)
        # 获得batch 次数，防止迭代器报错
        batch_num = len(encoder_inputs) // batch_size
        for b in range(batch_num):
            encoder_inputs, decoder_target, decoder_input = next(data_with_batch)
        
            feed_dict = {inputs: encoder_inputs, decoder_inputs: decoder_input,
                         targets: np.array(decoder_target),
                         }
            _, a, b, c = sess.run([train_op, loss, decoder_logits, labels_target], feed_dict)
            saver.save(sess, "model/my-model", global_step=ep)
            predict = sess.run(decoder_logits_argmax, feed_dict)
            accuracy = 1
            #print(_, a, b, c, predict)
            #print(predict)
            print(count)
            count += 1
    a = predict
    print(inference_layer([encoder_inputs[0]]))
    # ！！！！！！！！！训练时候需要保存字到数字的字典，不然预测错误

    
