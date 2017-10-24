#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import numpy as np
from six.moves import cPickle as pickle
import tensorflow as tf


''' bi2Embedding '''
class Embedding:
    SAVE_PATH = r'../data/text8_bi_embedding_cbow.pickle'   # 数据的存储路径


    def __init__(self):
        pass


    ''' 加载数据 '''
    @staticmethod
    def load():
        print '\nloading embedding from %s' % Embedding.SAVE_PATH
        with open(Embedding.SAVE_PATH, 'rb') as f:
            embeddings = pickle.load(f)
        print 'Finish loading'
        return embeddings


'''
 数据模块
'''
class Data:
    VOCABULARY_SIZE = 27 * 27               # 设置词典的数据量
    SAVE_PATH = '../data/text8_bi.pickle'   # 数据的存储路径

    def __init__(self, start_ratio = 0.0, end_ratio = 1.0):
        self.__load()  # 加载数据

        start_ratio = min(max(0.0, start_ratio), 1.0)       # 开始比例 必须在 0.0 - 1.0 之间
        end_ratio = min(max(0.0, end_ratio), 1.0)           # 结束比例 必须在 0.0 - 1.0 之间

        start_index = int(self.__dataLen * start_ratio)     # 数据的开始位置
        end_index = int(self.__dataLen * end_ratio)         # 数据的结束位置
        self.__data = self.__data[start_index: end_index]   # 根据 开始位置、结束位置 取数据
        self.__dataLen = len(self.__data)                   # 重置 数据的长度

        # 初始化 batch 相关数据
        self.__batchSize = 0
        self.__numUnRollings = 0
        self.__cursor = []

        self.__lastBatch = []


    ''' 加载数据 '''
    def __load(self):
        print '\nloading data from %s' % self.SAVE_PATH
        with open(self.SAVE_PATH, 'rb') as f:
            self.__data = pickle.load(f)
        print 'Finish loading'
        self.__dataLen = len(self.__data)   # data 的数据量


    '''
      返回 batch_size 个输入序列
        输入序列的长度为 num_unrollings
        shape 为 (num_unrollings + 1) * batch_size * vocabulary_size
    '''
    def nextBatches(self, batch_size, num_unrollings):
        if not self.__batchSize or self.__batchSize != batch_size:
            self.__batchSize = batch_size
            self.__numUnRollings = num_unrollings

            segment = self.__dataLen // self.__batchSize
            self.__cursor = [ index * segment for index in range(self.__batchSize)]

            self.__lastBatch = self.__nextBatch(self.__batchSize)

        batches = [self.__lastBatch]
        for step in range(self.__numUnRollings):
            batches.append( self.__nextBatch(self.__batchSize) )  # 将单一的 batch 加入到 batches
        self.__lastBatch = batches[-1]
        return batches


    '''
      返回 单一一个 batch，batch 里的每个元素只含有单个字符
        因为 lstm 里的输入是时序或序列的数据；一次输入 应该是 一个序列
        比如 __nextBatch 产生的 batch 如下 ['he', 'ni', 'we']，这里一个batch 含有3次输入
        那么真正输入的 batches 应该为 ['hello world', 'nice job bo', 'well done b']
            由多个 单一batch ['he', 'ni', 'we'], ['ll', 'ce', 'll'], ['o ', ' j', ' d'] ... ['d', 'o', 'b'] 组成
    '''
    def __nextBatch(self, batch_size):
        batch = []
        for b in range(batch_size):
            batch.append(self.__data[self.__cursor[b]])
            self.__cursor[b] = (self.__cursor[b] + 1) % self.__dataLen
        return np.array(batch)


    ''' 返回数据集的大小 '''
    def getSize(self):
        return self.__dataLen


# o_data = Data()
# batches = o_data.nextBatches(5, 10)

# batches2 = o_data.nextBatches(5, 10)
#
# import numpy as np
# for batch in batches:
#     string = ''
#     for val in batch:
#         string += Download.id2Char(np.argmax(val))
#     print string
#
# for batch in batches2:
#     string = ''
#     for val in batch:
#         string += Download.id2Char(np.argmax(val))
#     print string
