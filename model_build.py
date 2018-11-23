'''

by xlc time:2018-11-23 11:09:56
'''
import sys
#sys.path.append('D:/svn_codes/source/public_fun')
import os
main_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
main_path = '/'.join(main_path.split('/')[:-1])
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self):
        self.lstm_hidden_num = 100
        self.words_class = 100
        self.input_max_length = 15
        self.decoder_input_max_length = 19
        self.batch_size = 10
        self.fully_connect = 32
        self.inputs = tf.placeholder(tf.int32, shape=(None, None), name='inputs')
        self.decoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name='decoder_inputs')
        self.target = tf.placeholder(tf.int32, shape=(None, None), name='targets')

    def build_embedding(self, units_num, embedding_length):
        # embedding 层
        embedding = tf.get_variable('embedding', [units_num, embedding_length])
        return embedding

    def build_lstm(self, lstm_size, keep_prob, layers_num, name='lstm'):
        # lstm层 build
        def get_a_lstm(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope(name):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_lstm(lstm_size, keep_prob) for i in range(layers_num)])
        return cell

    def build_input(self, input_index, embedding_size=None):
        # 输入层 build
        """
        - embeddinng_size: ListType Length>=2
        """
        embedding_layer = build_embedding(embedding_size[0], embedding_size[1])
        embedding_out = tf.nn.embedding_lookup(embedding_layer, input_index)
        return embedding_out

    def build_encoder(self, inputs):
        embedding_layer = self.build_embedding(self.words_class, 100)
        lstm_inputs = tf.nn.embedding_lookup(embedding_layer, inputs)
        self.encoder_layer = self.build_lstm(self.hidden_num, 0.9, 1)
        # initial
        initial_state = self.encoder_layer.zero_state(self.batch_size, tf.float32)
        encoder_lstm_outputs, encoder_final_state = \
                              tf.nn.dynamic_rnn(self.encoder_layer, lstm_inputs, dtype=tf.float32)
        encoder_final_c = encoder_final_state[0][0]
        return encoder_final_c

    def build_decoder(self):
        decoder_lstm_inputs = tf.nn.embedding_lookup(self.embedding_layer, decoder_inputs)
        self.decoder_layer = build_lstm(lstm_hidden_num, 0.9, 1)

        decoder_lstm_outputs, decoder_final_state = tf.nn.dynamic_rnn(self.decoder_layer, decoder_lstm_inputs,\
                                                                      initial_state=encoder_final_state,\
                                                                      scope='plain_decoder')
        
    def build_loss(self, decoder_lstm_outputs):
        decoder_logits_argmax = tf.argmax(decoder_lstm_outputs, 2)
        decoder_logits = tf.contrib.layers.linear(decoder_lstm_outputs, self.fully_connect)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(targets, depth=fully_connect, dtype=tf.float32),
            logits=decoder_logits)
        loss = tf.reduce_mean(cross_entropy)
        self.train_op = tf.train_AdamOptimizer().minimize(loss)

    def inference_layer(inputs_infer):
        """
        推理层构建
        - 结束条件
          1. 达到最大字符数
          2. 输出<END>
        """
        global dictionary
        #global encoder_final_state
        stop_condition = False
        count = 0
        # 编码
        inputs_encode = tf.nn.embedding_lookup(self.embedding_layer, inputs_infer)
        encoder_lstm_outputs, encoder_final_state = tf.nn.dynamic_rnn(self.encoder_layer, inputs_encode, dtype=tf.float32)
        # 推断初始输入
        inference_initial_inputs = tf.nn.embedding_lookup(self.embedding_layer, [[dictionary[0]['bos']]])
        string = ''
        while not stop_condition:
            # 预测字符
            #判断是否终止
            #需要向量转字符
            print(11)
            inference_outputs, infenrence_state = tf.nn.dynamic_rnn(self.decoder_layer, inference_initial_inputs, \
                                                                    initial_state=encoder_final_state,scope="inference_decoder")
            outputs_str = dictionary[1][np.argmax(inference_outputs)]
            string += outputs_str
            if outputs_str == 'eos' or count > 20:
                stop_condition = True
            else:
                inference_initial_inputs = inference_outputs
                encoder_final_state = infenrence_state
            count += 1
            print(string)
        return string

    def model_train(self):
        pass

    def model_test(self):
        pass
    
if __name__ == '__main__':
    pass
