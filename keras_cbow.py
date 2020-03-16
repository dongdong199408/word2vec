#! -*- coding:utf-8 -*-
# Keras 2.0.6 ＋ Tensorflow 测试通过

import numpy as np
from keras.layers import Input, Embedding, Lambda, LSTM, Dense,concatenate
from keras.models import Model, load_model
from keras.utils import plot_model
import keras.backend as K
import jieba
import pandas as pd
import os
import keras
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau


# 每隔1000个epoch，学习率减小为原来的1/10
def scheduler(epoch):
    if epoch % 1000 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


# reduce_lr = LearningRateScheduler(scheduler)
def stopwordslist():  # 设置停用词
    stopwords = []
    if not os.path.exists('./stopwords.txt'):
        print('未发现停用词表！')
    else:
        stopwords = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


def getdata(fname):
    f = open(fname, 'r', encoding='UTF-8')
    lines = f.readlines()
    sentences = []
    data = []
    stopwords = stopwordslist()
    for line in lines:
        data.append(line.strip())  # 原始句子
        sts = list(jieba.cut(line.strip(), cut_all=False))  # 分词后
        splits = []  # 去停用词后
        for w in sts:
            if w not in stopwords:
                splits.append(w)
        sentences.append(splits)
    f.close()
    return data, sentences


def bulid_dic(sentences):  # 建立各种字典
    words = {}  # 词频表
    nb_sentence = 0  # 总句子数
    total = 0.  # 总词频

    for d in sentences:
        nb_sentence += 1
        for w in d:
            if w not in words:
                words[w] = 0
            words[w] += 1
            total += 1
        if nb_sentence % 100 == 0:
            print(u'已经找到%s个句子' % nb_sentence)

    words = {i: j for i, j in words.items() if j >= min_count}  # 截断词频
    id2word = {i + 1: j for i, j in enumerate(words)}  # id到词语的映射，0表示UNK
    id2word.update({0:'UNK'})
    word2id = {j: i for i, j in id2word.items()}  # 词语到id的映射
    nb_word = len(words) + 1  # 总词数（算上填充符号0）

    subsamples = {i: j / total for i, j in words.items() if j / total > subsample_t}
    subsamples = {i: subsample_t / j + (subsample_t / j) ** 0.5 for i, j in
                  subsamples.items()}  # 这个降采样公式，是按照word2vec的源码来的
    subsamples = {word2id[i]: j for i, j in subsamples.items() if j < 1.}  # 降采样表
    return nb_sentence, id2word, word2id, nb_word, subsamples


def data_generator(word2id, subsamples, data):  # 训练数据生成器
    x, y = [], []
    _ = 0
    for d in data:
        d = [0] * window + [word2id[w] for w in d if w in word2id] + [0] * window
        r = np.random.random(len(d))
        for i in range(window, len(d) - window):
            if d[i] in subsamples and r[i] > subsamples[d[i]]:  # 满足降采样条件的直接跳过
                continue
            x.append(d[i - window:i] + d[i + 1:i + 1 + window])
            y.append([d[i]])
        _ += 1
        if _ == nb_sentence_per_batch:
            x, y = np.array(x), np.array(y)
            z = np.zeros((len(x), 1))
            c=np.random.randint(0,5,size=(len(x), 1),dtype='int32')
            return [x, y], [z,c]


def build_w2vm(word_size, window, nb_word, nb_negative):
    K.clear_session()  # 清除之前的模型，省得压满内存
    # CBOW输入
    target_word = Input(shape=(1,), dtype='int32',name='target_input')
    input_words = Input(shape=(window * 2,), dtype='int32', name='main_input')
    #首先定义一个分类的模型

    #input_length: 输入序列的长度，当它是固定的时。 如果你需要连接 Flatten 和 Dense 层，则这个参数是必须的 （没有它，dense 层的输出尺寸就无法计算）。
    input_vecs = Embedding(nb_word, word_size, name='word2vec_main')(input_words)
    class_input_vecs = Embedding(nb_word, word_size, name='word2vec_target')(target_word)
    # 自定义一个lstm层
    lstm_out = LSTM(32)(class_input_vecs)
    dense_out = Dense(64, activation='relu')(lstm_out)
    # 增加一个全连接
    class_output = Dense(5, activation='sigmoid', name='class_out')(dense_out)
    #全面函数层用lambda定义
    input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs)  # CBOW模型，直接将上下文词向量求和,也可去平均
    # 构造随机负样本，与目标组成抽样

    target_input=concatenate([input_vecs_sum ,class_output],name='concat_class')

    #生成均匀分布负样本，从16个样本中，一个正样本找到答案，应该打乱顺序的找
    negatives = Lambda(lambda x: K.random_uniform((K.shape(x)[0], nb_negative), 0, nb_word, 'int32'),name='target_negatives')(target_word)
    samples = Lambda(lambda x: K.concatenate(x,axis=-1),name='target_samples')([target_word, negatives])  # 构造抽样，负样本随机抽。负样本也可能抽到正样本，但概率小。
    # 只在抽样内做Dense和softmax
    #相当于添加权重曾
    softmax_weights = Embedding(nb_word, word_size+5, name='W')(samples)
    softmax_biases = Embedding(nb_word, 1, name='b')(samples)
    #相当于sampled_softmax_loss
    softmax = Lambda(lambda x:
                     K.softmax((K.batch_dot(x[0], K.expand_dims(x[1], 2)) + x[2])[:, :, 0])
                     ,name='main_out')([softmax_weights, target_input, softmax_biases])  # 用Embedding层存参数，用K后端实现矩阵乘法，以此复现Dense层的功能
    #留意到，我们构造抽样时，把目标放在了第一位，也就是说，softmax的目标id总是0，这可以从data_generator中的z变量的写法可以看出

    model = Model(inputs=[input_words, target_word], outputs=[softmax,class_output])
    model.compile(loss={'main_out':'sparse_categorical_crossentropy', 'class_out':'sparse_categorical_crossentropy'},
                  loss_weights={'main_out': 1, 'class_out': 0.5},
                        optimizer='adam', metrics=['accuracy'])
    '''
    如果你的 targets 是 one-hot 编码，用 categorical_crossentropy
　　one-hot 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]
    如果你的 tagets 是 数字编码 ，用 sparse_categorical_crossentropy
　　数字编码：2, 0, 1
    '''
    # 请留意用的是sparse_categorical_crossentropy而不是categorical_crossentropy
    #model.summary()
    return model


def most_similar(word2id, w, k=10):  # 找相似性Topk个词
    # 通过词语相似度，检查我们的词向量是不是靠谱的
    model = load_model('./model/word2vec.h5')  # 载入模型 在数据集较大的时候用时间换空间
    # weights = model.get_weights()#可以顺便看看每层的权重
    # for wei in weights:
    #     print(wei.shape)
    embeddings = model.get_weights()[0]
    normalized_embeddings = embeddings / (embeddings ** 2).sum(axis=1).reshape((-1, 1)) ** 0.5  # 词向量归一化，即将模定为1
    v = normalized_embeddings[word2id[w]]
    sims = np.dot(normalized_embeddings, v)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[i], sims[i]) for i in sort[:k]]

#保存embedding文件
def save_embedding(final_embeddings, model_path, reverse_dictionary):
    f = open(model_path,'w+')
    for index, item in enumerate(final_embeddings):
        f.write(reverse_dictionary[index] + '\t' + ','.join([str(vec) for vec in item]) + '\n')
    f.close()

if __name__ == '__main__':
    fname = './result/chatrobot_messageresult_sentence.txt'  # 数据集(语料库) 一个文档
    log_filepath = '/tmp/keras_log'
    word_size = 200  # 词向量维度
    window = 5  # 窗口大小
    nb_negative = 150  # 随机负采样的样本数
    min_count = 0  # 频数少于min_count的词将会被抛弃
    nb_worker = 4  # 读取数据的并发数
    nb_epoch = 2  # 迭代次数，由于使用了adam，迭代次数1～2次效果就相当不错
    subsample_t = 1e-5  # 词频大于subsample_t的词语，会被降采样，这是提高速度和词向量质量的有效方案
    nb_sentence_per_batch = 20
    # 目前是以句子为单位作为batch，多少个句子作为一个batch（这样才容易估计训练过程中的steps参数，另外注意，样本数是正比于字数的。）

    data, sentences = getdata(fname)  # 读原始数据
    nb_sentence, id2word, word2id, nb_word, subsamples = bulid_dic(sentences)  # 建字典
    ipt, opt = data_generator(word2id, subsamples, data)  # 构造训练数据
    model = build_w2vm(word_size, window, nb_word, nb_negative)  # 搭模型
    #tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    # 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的权值，每层输出值的分布直方图
    # reduce_lr = LearningRateScheduler(scheduler)

    '''
    第二种学习率下降
    monitor：被监测的量
    factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
    patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
    epsilon：阈值，用来确定是否进入检测值的“平原区”
    cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
    min_lr：学习率的下限
    '''
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.0001, mode='auto')
    #cbks = [tb_cb]
    model.fit(ipt, opt,
              steps_per_epoch=int(nb_sentence / nb_sentence_per_batch),
              epochs=nb_epoch,
              callbacks = [reduce_lr]
              #callbacks=cbks#,
              # workers=nb_worker,
              # use_multiprocessing=False
              )
    model.save('./model/word2vec.h5')
    print('model saved!')
    model = load_model('./model/word2vec.h5')
    plot_model(model, to_file='model.png')
    embeddings = model.get_weights()[0]
    save_embedding(embeddings,'./model/keras_cbow_wordvec.bin',id2word)