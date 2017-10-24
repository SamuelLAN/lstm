#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import random
import collections
import numpy as np
from six.moves import cPickle as pickle


'''
 将数据转换为 bi-level 形式存储
'''
class TransFormBi:
    DATA_ROOT = '../data'
    ORIGIN_PATH = os.path.join(DATA_ROOT, 'text8.pickle')
    SAVE_PATH = os.path.join(DATA_ROOT, 'text8_bi.pickle')

    def __init__(self):
        pass


    def run(self):
        if os.path.isfile(TransFormBi.SAVE_PATH):
            print 'exist %s, done' % TransFormBi.SAVE_PATH
            return

        self.__load()           # 加载数据
        self.__transform2Bi()   # 将数据转换为 bi-level 形式
        self.__saveBiData()     # 保存数据


    ''' 加载数据 '''
    def __load(self):
        print '\nloading data from %s ...' % TransFormBi.ORIGIN_PATH
        with open(TransFormBi.ORIGIN_PATH, 'rb') as f:
            self.__dataOrigin = pickle.load(f)
        print 'Finish loading'


    ''' 将数据转换为 bi-level '''
    def __transform2Bi(self):
        print '\nTransforming data to bi-level ...'
        self.__data = []
        bi = []

        for character in self.__dataOrigin:
            bi.append(character)
            if len(bi) == 2:
                self.__data.append(bi[0] * 27 + bi[1])
                bi = []
        del self.__dataOrigin

        if bi and len(bi) == 1 and bi[0] != ' ':
            bi.append(' ')
            self.__data.append(bi)

        print 'Finish transforming'


    ''' 保存 bi-level 数据 '''
    def __saveBiData(self):
        print '\nSaving %s ... ' % TransFormBi.SAVE_PATH
        with open(TransFormBi.SAVE_PATH, 'wb') as f:
            pickle.dump(self.__data, f, pickle.HIGHEST_PROTOCOL)
        print 'Finish saving'


    ''' 将 id 转换为 bi-level 的字符 '''
    @staticmethod
    def id2Bi(_id):
        first = int(_id / 27)
        second = int(_id % 27)

        def __id2char(_id_char):
            return chr(96 + _id_char) if _id_char != 0 else ' '

        return __id2char(first) + __id2char(second)


'''
 数据模块
'''
class Data:

    def __init__(self, start_ratio = 0.0, end_ratio = 1.0):
        self.__load()   # 加载数据

        start_ratio = min(max(0.0, start_ratio), 1.0)  # 开始比例 必须在 0.0 - 1.0 之间
        end_ratio = min(max(0.0, end_ratio), 1.0)  # 结束比例 必须在 0.0 - 1.0 之间

        start_index = int(self.__dataLen * start_ratio)  # 数据的开始位置
        end_index = int(self.__dataLen * end_ratio)  # 数据的结束位置
        self.__data = self.__data[start_index: end_index]  # 根据 开始位置、结束位置 取数据
        self.__dataLen = len(self.__data)  # 重置 数据的长度

        # 初始化变量
        self.__windowSize = 1                               # 目标词 左边 或 右边的 上下文词语数量
        self.__numOfContext = self.__windowSize * 2         # 上下文的词语数量为 skip_window 的两倍
        self.__dataIndex = -1                               # 目前处于 data 的所在位置 index


    ''' 加载数据 '''
    def __load(self):
        with open(TransFormBi.SAVE_PATH, 'rb') as f:
            self.__data = pickle.load(f)
        self.__dataLen = len(self.__data)                   # data 的数据量


    '''
      设置 window_size ；没有设置时，默认为 1
        window_size 指：目标词 左边 或 右边的 上下文词语数量
          如：一句话 ['Today', 'is', 'a', 'nice', 'day', 'and', 'I', 'want', 'to', 'go', 'sighting']
          假如 目标词为 'nice'，
            若 window_size 为 1, 则 'nice' 的上下文为 ['a', 'day'] ;
            若 window_size 为 2, 则 'nice' 的上下文为 ['is', 'a', 'day', 'and'] ;
    '''
    def setWindowSize(self, window_size):
        self.__windowSize = min(int(window_size), 1)
        self.__numOfContext = 2 * self.__windowSize         # 上下文的词语数量为 skip_window 的两倍


    ''' Skip-gram 模型时，获取下一 batch 的数据 '''
    def skipGramNextBatch(self, batch_size, loop = True):
        assert batch_size % self.__numOfContext == 0        # batch_size 的大小必须为 上下文的整数倍
        if not loop and self.__dataIndex >= self.__dataLen - self.__windowSize:
            self.__dataIndex = 0
            return None, None

        batch = []
        label = []
        span = 2 * self.__windowSize + 1                    # [ skip_window target skip_window ]

        if self.__dataIndex != -1:                          # 由于初始化 _buffer 时会额外增加 span 的位置，需要修正 data_index
            if self.__dataIndex >= span:
                self.__dataIndex -= span
            else:
                self.__dataIndex = self.__dataLen + self.__dataIndex - span
        else:
            self.__dataIndex = 0                            # 第一次使用 data_index，初始化为 0

        _buffer = collections.deque(maxlen=span)             # 初始化 _buffer
        for _ in range(span):                               # 用最开始位置的 span 个元素填满 _buffer
            _buffer.append(self.__data[self.__dataIndex])
            self.__dataIndex = (self.__dataIndex + 1) % self.__dataLen

        for i in range(batch_size // self.__numOfContext):
            target = self.__windowSize                      # 目标词在 _buffer 中的所在位置
            targets_to_avoid = [ self.__windowSize ]        # 取上下文时，必须避开的位置；初始值为目标词汇的位置

            # 将一个 window 里的目标词汇与上下文，分别加入 batch 和 label
            for j in range(self.__numOfContext):

                # 选取上下文词汇，此时的 target 为上下文词汇在 _buffer 中的位置；不能选中目标词，且不能选中已经选过的同一个 window 里的词
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)             # 将当前将要取的 上下文词汇 加入 targets_to_avoid，避免重复取

                batch.append(_buffer[self.__windowSize])     # 目标词汇始终位于 window 里的中间位置；window_size * 2 + 1 == window 的大小
                label.append(_buffer[target])                # 将上下文词汇加入到 label

            # 取完一个 window 里的数据，window 往右移动一位
            _buffer.append(self.__data[self.__dataIndex])
            self.__dataIndex = (self.__dataIndex + 1) % self.__dataLen

            if not loop and self.__dataIndex >= self.__dataLen - self.__windowSize:
                break

        batch = np.array(batch)
        label = np.array(label).reshape((batch_size, 1))
        return batch, label


    ''' CBOW 模型时，获取下一 batch 的数据 '''
    def cBOWNextBatch(self, batch_size, loop = True):
        assert batch_size <= self.__dataLen - self.__windowSize * 2 # batch_size 最大不能超过 数据的总量 减 上写文词汇的个数
        if not loop and self.__dataIndex >= self.__dataLen - self.__windowSize:
            self.__dataIndex = 0
            return None, None

        batch = []
        label = []
        span = 2 * self.__windowSize + 1  # [ skip_window target skip_window ]

        if self.__dataIndex != -1:  # 由于初始化 _buffer 时会额外增加 span 的位置，需要修正 data_index
            if self.__dataIndex >= span:
                self.__dataIndex -= span
            else:
                self.__dataIndex = self.__dataLen + self.__dataIndex - span
        else:
            self.__dataIndex = 0  # 第一次使用 data_index，初始化为 0

        _buffer = collections.deque(maxlen=span)  # 初始化 _buffer
        for _ in range(span):  # 用最开始位置的 span 个元素填满 _buffer
            _buffer.append(self.__data[self.__dataIndex])
            self.__dataIndex = (self.__dataIndex + 1) % self.__dataLen

        for i in range(batch_size):
            batch.append([j for index, j in enumerate(_buffer) if index != self.__windowSize])
            label.append(_buffer[self.__windowSize])

            # 取完一个 window 里的数据，window 往右移动一位
            _buffer.append(self.__data[self.__dataIndex])
            self.__dataIndex = (self.__dataIndex + 1) % self.__dataLen

            if not loop and self.__dataIndex >= self.__dataLen - self.__windowSize:
                break

        batch = np.array(batch).reshape((batch_size, span - 1))
        label = np.array(label).reshape((batch_size, 1))
        return batch, label

