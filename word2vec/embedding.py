#!/usr/bin/Python
# -*- coding: utf-8 -*-
import base
import load
import random
import numpy as np
import tensorflow as tf
from matplotlib import pylab
from six.moves import cPickle as pickle


'''
 CBow 与 Skip-gram 模型的基类
'''
class Embedding(base.NN):
    MODEL_NAME = 'embedding'    # 模型的名称(默认使用 Cbow 模型)

    EPOCH_TIMES = 200           # 迭代的 epoch 次数
    BATCH_SIZE = 128            # 随机梯度下降的 batch 大小
    BASE_LEARNING_RATE = 1.0    # 初始学习率

    SKIP_WINDOW = 1             # 目标词 左边 或 右边的 上下文词语数量
    NUM_NEG_SAMPLE = 64         # 负样本个数

    VALID_SIZE = 16             # 验证集大小

    VOCABULARY_SIZE = 27 * 27   # 设置词典的数据量
    EMBEDDING_SIZE = 64         # 词向量的大小
    # 权重矩阵的 shape
    SHAPE = [VOCABULARY_SIZE, EMBEDDING_SIZE]

    TRAIN_DATA_START_RATIO = 0.0
    TRAIN_DATA_END_RATIO = 0.64

    # 模型存储的路径，默认使用 cbow 模型
    SAVE_PATH = r'../data/text8_bi_embedding.pickle'


    ''' 初始化 X 与 y 的接口 （ 被 self.init() 调用 ） '''
    def initXY(self):
        pass


    ''' 初始化接口 （ __init__ 函数会自动调用 ） '''
    def init(self):
        # 加载数据
        self.load()

        self.initXY()

        # 从 vocabulary 中随机采样 valid_size 个样本作为验证集的样本
        self.valExample = np.array(random.sample(range(self.VOCABULARY_SIZE), self.VALID_SIZE))
        self.__valX = tf.constant(self.valExample, dtype=tf.int32)  # 验证集数据

        # 常量
        self.iterPerEpoch = int(self.VOCABULARY_SIZE // self.BATCH_SIZE)
        self.steps = self.EPOCH_TIMES * self.iterPerEpoch

        # 随训练次数增多而衰减的学习率
        self.learningRate = self.getLearningRate(
            self.BASE_LEARNING_RATE, self.globalStep, self.BATCH_SIZE, self.steps, self.DECAY_RATE
        )

        self.normalizeEmbedding = np.array([])  # 初始化最终的词向量


    ''' 加载数据 '''
    def load(self):
        self.data = load.Data(self.TRAIN_DATA_START_RATIO, self.TRAIN_DATA_END_RATIO)


    ''' 计算 loss，采取 负采样(negative sampling) 的方法 '''
    def getLoss(self, weight, bias, input_embed, labels):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=weight, biases=bias, inputs=input_embed, labels=labels,
                                           num_sampled=self.NUM_NEG_SAMPLE, num_classes=self.VOCABULARY_SIZE)
            )


    ''' 获取 train_op '''
    def getTrainOp(self, loss, learning_rate, global_step):
        tf.summary.scalar('loss', loss)  # 记录 loss 到 TensorBoard

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdagradOptimizer(learning_rate)
            return optimizer.minimize(loss, global_step=global_step)


    ''' 计算相似度 '''
    def calSimilarity(self, embeddings):
        with tf.name_scope('similarity'):
            # 只在 y 轴上求 平方和 然后 开根，x 轴维数不变 ( x 轴对应的是 batch 的数量 )
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

            # normalize 当前的 embedding 参数
            self.normalizeEmbedding = embeddings / norm

            # 在当前的 embedding 参数中计算 验证集的词向量
            valid_embeddings = tf.nn.embedding_lookup(self.normalizeEmbedding, self.__valX)

            self.similarity = tf.matmul(valid_embeddings, tf.transpose(self.normalizeEmbedding), name='similarity')


    ''' 将 tsne 降维后的数据 画成 2d 图像 '''
    def plot2dEmbeddingSim(self, embeddings, num_words_to_plot):
        self.echo('plotting 2d embeddings image by tsne ...')

        self.echo('Doing tsne ...')
        words_to_plot = [load.TransFormBi.id2Bi(i) for i in range(1, num_words_to_plot + 1)]  # 第 0 个为 UNK
        embeddings = self.tsne(embeddings, num_words_to_plot)
        self.echo('finish tsne')

        assert embeddings.shape[0] >= len(words_to_plot)  # embeddings 的维度需要比 label 的词数量多
        pylab.figure(figsize=(15, 15))
        for i, word in enumerate(words_to_plot):
            x, y = embeddings[i, :]
            pylab.scatter(x, y)
            pylab.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

        pylab.show()

        self.echo('Finish')


    ''' 保存 embeddings 到文件 '''
    def saveEmbedding(self, embeddings):
        self.echo('\nSaving %s ... ' % self.SAVE_PATH)
        with open(self.SAVE_PATH, 'wb') as f:
            pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)
        self.echo('Finish saving')
