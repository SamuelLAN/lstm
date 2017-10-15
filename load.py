#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import string
import random
import zipfile
import collections
import numpy as np
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


class Download:
    URL = 'http://mattmahoney.net/dc/'  # 数据的下载目录
    FILE_NAME = 'text8.zip'             # 文件名
    FILE_SIZE = 31344016                # 字节
    DATA_ROOT = 'data'                  # 数据存放的目录

    VOCABULARY_SIZE = len(string.ascii_lowercase) + 1   # [a-z] + ' '
    ORD_FIRST_LETTER = ord(string.ascii_lowercase[0])   # a 的 ascii 码

    SAVE_PATH = os.path.join(DATA_ROOT, 'text8.pickle') # 保存数据的文件路径

    def __init__(self):
        pass


    ''' 主函数 '''
    @staticmethod
    def run():
        if os.path.isfile(Download.SAVE_PATH):
            print 'Already exist data in %s\nDone' % Download.SAVE_PATH
            return

        file_path = Download.__maybeDownload()

        chars = Download.__readData(file_path)

        Download.__saveData(chars)


    ''' 下载数据 '''
    @staticmethod
    def __maybeDownload():
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.isdir(Download.DATA_ROOT):           # 若 data 目录不存在，创建 data 目录
            os.mkdir(Download.DATA_ROOT)
        file_path = os.path.join(Download.DATA_ROOT, Download.FILE_NAME)

        if os.path.exists(file_path):                       # 若已存在该文件
            statinfo = os.stat(file_path)
            if statinfo.st_size == Download.FILE_SIZE:      # 若该文件正确，直接返回 file_path
                print('Found and verified %s' % file_path)
                return file_path
            else:                                           # 否则，删除文件重新下载
                os.remove(file_path)

        download_url = Download.URL + Download.FILE_NAME
        print('Downloading %s ...' % download_url)
        filename, _ = urlretrieve(download_url, file_path)  # 下载数据
        print('Finish downloading')

        statinfo = os.stat(filename)
        if statinfo.st_size == Download.FILE_SIZE:          # 校验数据是否正确下载
            print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser ?')
        return filename


    ''' 读取压缩文件 '''
    @staticmethod
    def __readData(file_path):
        """Extract the first file enclosed in a zip file as a list of words"""
        with zipfile.ZipFile(file_path) as f:
            tmp_path = f.namelist()[0]                      # 压缩文件里只有一个 'text8' 文件
            print '\nReading %s/%s' % (file_path, tmp_path)
            content = f.read(tmp_path)                      # 读取文件内容
            print 'Finish Reading %s/%s' % (file_path, tmp_path)

        print '\nTransferring data format ...'
        content = content.lower()                           # 保证全部字符转为小写
        content = [Download.char2Id(i) for i in content]    # 将字符全部转为 id
        print 'Finish transferring'
        return content


    ''' 保存数据 '''
    @staticmethod
    def __saveData(data):
        print '\nSaving %s ... ' % Download.SAVE_PATH
        with open(Download.SAVE_PATH, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print 'Finish saving'


    ''' 将字符转为 id '''
    @staticmethod
    def char2Id(char):
        if char in string.ascii_lowercase:
            return ord(char) - Download.ORD_FIRST_LETTER + 1
        elif char == ' ':
            return 0
        else:
            print 'Unexpected character: %s' % char
            return 0


    ''' 将 id 转为 字符 '''
    @staticmethod
    def id2Char(_id):
        if _id == 0:
            return ' '
        else:
            return chr(96 + _id)


class Data:

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
        print '\nloading data from %s' % Download.SAVE_PATH
        with open(Download.SAVE_PATH, 'rb') as f:
            self.__data = pickle.load(f)
        print 'Finish loading'
        self.__dataLen = len(self.__data)                   # data 的数据量


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
            batches.append(self.__nextBatch(self.__batchSize))
        self.__lastBatch = batches[-1]
        return batches


    '''
      返回 单一一个 batch，batch 里的每个元素只含有单个字符
        因为 lstm 里的输入是时序或序列的数据；一次输入 应该是 一个序列
        比如 __nextBatch 产生的 batch 如下 ['h', 'n', 'w']，这里一个batch 含有3次输入
        那么真正输入的 batches 应该为 ['hello world', 'nice job bo', 'well done b']
            由多个 单一batch ['h', 'n', 'w'], ['e', 'i', 'e'], ['l', 'c', 'l'] ... ['d', 'o', 'b'] 组成
    '''
    def __nextBatch(self, batch_size):
        batch = np.zeros(shape=(batch_size, Download.VOCABULARY_SIZE), dtype=np.float)
        for b in range(batch_size):
            batch[b, self.__data[self.__cursor[b]]] = 1.0
            self.__cursor[b] = (self.__cursor[b] + 1) % self.__dataLen
        return batch


    ''' 返回数据集的大小 '''
    def getSize(self):
        return self.__dataLen


# Download.run()
#
# o_data = Data()
# batches = o_data.nextBatches(5, 10)
