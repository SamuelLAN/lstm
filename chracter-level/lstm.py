#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
root_dir_path = os.path.split(cur_dir_path)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(root_dir_path)

import base
import load
import random
import numpy as np
import tensorflow as tf


'''
 长短途记忆模型 LSTM
    一个比较简单易懂的讲解：http://blog.csdn.net/prom1201/article/details/52221822
'''
class LSTM(base.NN):
    MODEL_NAME = 'lstm_chracter_level'                             # 模型的名称

    BASE_LEARNING_RATE = 10.0                       # 初始 学习率
    DECAY_RATE = 1e-3                                # 学习率 的 下降速率

    CLIP_NORM = 1.25                                # 梯度剪切的参数

    NUM_UNROLLINGS = 10                             # 序列数据一次输入 nul_unrollings 个数据
    BATCH_SIZE = 64                                 # 随机梯度下降的 batch 大小
    EPOCH_TIMES = 20                                # 迭代的 epoch 次数

    VOCABULARY_SIZE = load.Download.VOCABULARY_SIZE # 词典大小
    NUM_NODES = 64                                  # layer1 有多少个节点

    SHAPE_DATA = [BATCH_SIZE, VOCABULARY_SIZE]      # 输入 data 的 shape
    SHAPE_INPUT = [VOCABULARY_SIZE, NUM_NODES]      # cell 中与输入相乘的权重矩阵的 shape
    SHAPE_OUTPUT = [NUM_NODES, NUM_NODES]           # cell 中与上次输出相乘的权重矩阵的 shape

    # REGULAR_BETA = 0.01                             # 正则化的 beta 参数
    # MOMENTUM = 0.9                                  # 动量的大小

    # 若校验集的 perplexity 连续超过 EARLY_STOP_CONDITION 次没有低于 best_perplexity_val, 则提前结束迭代
    EARLY_STOP_CONDITION = 100


    ''' 自定义 初始化变量 过程 '''
    def init(self):
        # 加载数据
        self.load()

        # 常量
        self.__iterPerEpoch = int(self.__trainSize // self.BATCH_SIZE)

        self.__steps = self.__iterPerEpoch
        self.__summaryFrequency = 100
        # self.__iterPerEpoch = int(self.__trainSize // self.BATCH_SIZE)
        # self.__steps = self.EPOCH_TIMES * self.__iterPerEpoch
        #
        # # TODO 删除该提示
        # print 'iter_per_epoch: %d' % self.__iterPerEpoch
        # print 'steps: %d' % self.__steps

        # 输入 与 label
        self.__trainData = list()
        for _ in range(self.NUM_UNROLLINGS + 1):
            self.__trainData.append( tf.placeholder(tf.float32, shape=self.SHAPE_DATA, name='input') )
        self.__X = self.__trainData[:self.NUM_UNROLLINGS]
        self.__y = self.__trainData[1:]                       # label 为 输入数据往后移一位 (序列数据)

        # 随训练次数增多而衰减的学习率
        self.__learningRate = self.getLearningRate(
            self.BASE_LEARNING_RATE, self.globalStep, self.__steps / 10, self.DECAY_RATE
        )


    ''' 初始化 各种门需要的 权重矩阵 以及 偏置量 '''
    @staticmethod
    def __initGate(name=''):
        # 与输入相乘的权重矩阵
        w_x = LSTM.initWeight(LSTM.SHAPE_INPUT, name=name + 'x')
        # 与上次输出相乘的权重矩阵
        w_m = LSTM.initWeight(LSTM.SHAPE_OUTPUT, name=name + 'm')
        # bias
        b = LSTM.initBias(LSTM.SHAPE_INPUT, name=name + 'b')
        return w_x, w_m, b


    ''' 初始化权重矩阵 '''
    @staticmethod
    def initWeight(shape, name='weights'):
        return tf.Variable(tf.truncated_normal(shape, -0.1, 0.1), name=name)


    ''' 加载数据 '''
    def load(self):
        self.__trainSet = load.Data(0.0, 0.64)          # 按 0.64 的比例划分训练集
        self.__valSet = load.Data(0.64, 0.8)            # 按 0.16 的比例划分校验集
        self.__testSet = load.Data(0.8)                 # 按 0.2  的比例划分测试集

        self.__trainSize = self.__trainSet.getSize()
        self.__valSize = self.__valSet.getSize()
        self.__testSize = self.__testSet.getSize()


    ''' 模型 '''
    def model(self):
        # ********************** 初始化模型所需的变量 **********************

        # with tf.name_scope('cell'):

        # 输入门
        with tf.name_scope('input_gate'):
            self.__wIX, self.__wIM, self.__bI = self.__initGate()

        # 忘记门
        with tf.name_scope('forget_gate'):
            self.__wFX, self.__wFM, self.__bF = self.__initGate()

        # 记忆单元
        with tf.name_scope('lstm_cell'):
            self.__wCX, self.__wCM, self.__bC = self.__initGate()

        # 输出门
        with tf.name_scope('output_gate'):
            self.__wOX, self.__wOM, self.__bO = self.__initGate()

        # 在序列 unrollings 之间保存状态的变量
        self.__savedOutput = tf.Variable(tf.zeros([self.BATCH_SIZE, self.NUM_NODES]),
                                         trainable=False, name='saved_output')
        self.__savedState = tf.Variable(tf.zeros([self.BATCH_SIZE, self.NUM_NODES]),
                                        trainable=False, name='saved_state')

        # 最后的分类器
        w_shape = [self.NUM_NODES, self.VOCABULARY_SIZE]
        self.__w = self.initWeight(w_shape, name='w')
        self.__b = self.initBias(w_shape, name='b')

        # ************************* 生成模型 ****************************

        self.__outputs = list()
        output = self.__savedOutput
        state = self.__savedState
        for _input in self.__X:
            output, state = self.__cell(_input, output, state)
            self.__outputs.append(output)

        # *********************** 计算 loss *****************************

        # 保证状态传递完毕
        with tf.control_dependencies([self.__savedOutput.assign(output), self.__savedState.assign(state)]):
            self.__logits = tf.nn.xw_plus_b( tf.concat(self.__outputs, 0), self.__w, self.__b )     # 分类器
            self.getLoss()  # 计算 loss


    ''' 定义每个单元里的计算过程 '''
    def __cell(self, _input, _output, _state):
        with tf.name_scope('cell'):
            with tf.name_scope('input_gate'):
                input_gate = tf.sigmoid( tf.matmul(_input, self.__wIX) + tf.matmul(_output, self.__wIM) + self.__bI, name='input_gate' )

            with tf.name_scope('forget_gate'):
                forget_gate = tf.sigmoid( tf.matmul(_input, self.__wFX) + tf.matmul(_output, self.__wFM) + self.__bF, name='forget_gate' )

            update = tf.add( tf.matmul(_input, self.__wCX) + tf.matmul(_output, self.__wCM), self.__bC, name='update')

            _state = tf.add( forget_gate * _state, input_gate * tf.tanh(update), name='state' )

            with tf.name_scope('output_gate'):
                output_gate = tf.sigmoid( tf.matmul(_input, self.__wOX) + tf.matmul(_output, self.__wOM) + self.__bO, name='output_gate' )

            _output = tf.multiply(output_gate, tf.tanh(_state), name='output')

            return _output, _state


    ''' 计算 loss '''
    def getLoss(self):
        with tf.name_scope('loss'):
            self.__labels = tf.concat(self.__y, 0, name='labels')
            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.__logits, labels=self.__labels),
                name='loss'
            )


    ''' 获取预测值 '''
    def inference(self):
        with tf.name_scope('predict'):
            # lstm 的输出
            predict = tf.nn.softmax(self.__logits, name='predict')

        with tf.name_scope('perplexity'):
            # 计算 perplexity
            self.__perplexity = tf.exp(
                - tf.reduce_sum( tf.log(predict, name='log_predict') * self.__labels ) /
                tf.cast(tf.shape(self.__labels)[0], dtype=tf.float32), name='perplexity')

        # # 将 perplexity 记录到 summary 中
        # tf.summary.scalar('perplexity_train', self.__perplexity)


    ''' 获取 train_op '''
    def getTrainOp(self, loss, learning_rate, global_step):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.CLIP_NORM)
            return optimizer.apply_gradients(zip(gradients, v), global_step=global_step)


    ''' 记录训练过程中的 loss 与 perplexity 的均值 '''
    def __summaryMean(self):
        with tf.name_scope('summary'):
            # 新建两个 placeholder 用于 tensorboard 可以记录 loss 与 perplexity 的均值
            self.__lossPH = tf.placeholder(tf.float32, name='loss_ph')
            self.__perplexityPH = tf.placeholder(tf.float32, name='perplexity')

            # self.__perplexityTrain = tf.placeholder(tf.float32, name='perplexity_train_ph')
            # self.__perplexityVal = tf.placeholder(tf.float32, name='perplexity_val_ph')

            # 将数据记录到 tensorboard summary
            tf.summary.scalar('mean_loss', self.__lossPH)
            tf.summary.scalar('mean_perplexity', self.__perplexityPH)
            # tf.summary.scalar('mean_perplexity', self.__perplexityTrain)
            # tf.summary.scalar('validation_perplexity', self.__perplexityVal)


    ''' 初始化 校验集 或 测试集 预测所需的变量 '''
    def __predict(self):
        self.__sampleInput = tf.placeholder(tf.float32, shape=[1, self.VOCABULARY_SIZE],
                                            name='sample_input')

        self.__sampleSavedOutput = tf.Variable(tf.zeros([1, self.NUM_NODES]), name='sample_saved_output')
        self.__sampleSavedState = tf.Variable(tf.zeros([1, self.NUM_NODES]), name='sample_saved_state')
        self.__sampleLogProb = tf.Variable(0, dtype=tf.float32, name='sample_log_prob')

        self.__sampleResetState = tf.group(
            self.__sampleSavedOutput.assign(tf.zeros([1, self.NUM_NODES])),
            self.__sampleSavedState.assign(tf.zeros([1, self.NUM_NODES])),
            self.__sampleLogProb.assign(0)
        )

        self.__sampleOutput, self.__sampleState = self.__cell(self.__sampleInput, self.__sampleSavedOutput,
                                                              self.__sampleSavedState)

        with tf.control_dependencies([
            self.__sampleSavedOutput.assign(self.__sampleOutput),
            self.__sampleSavedState.assign(self.__sampleState)
        ]):
            self.__samplePrediction = tf.nn.softmax(tf.nn.xw_plus_b(self.__sampleOutput, self.__w, self.__b))


    ''' 评估 dataset 的 perplexity '''
    def __evaluate(self, dataset, data_size):
        self.sess.run(self.__sampleResetState)
        valid_logprob = 0
        for _ in range(data_size):
            batch = dataset.nextBatches(1, 1)
            predictions = self.sess.run(self.__samplePrediction, {self.__sampleInput: batch[0]})
            valid_logprob += self.__logProb(predictions, batch[1])
        return float(np.exp(valid_logprob / data_size))


    ''' 随机预测文本 '''
    def __randomPredictText(self, lines = 5):
        self.echo('=' * 80)
        for _ in range(lines):
            feed = self.__sample(self.__randomDistribution())
            sentence = self.matrix2char(feed)
            self.sess.run(self.__sampleResetState)
            for _ in range(79):
                prediction = self.sess.run(self.__samplePrediction, {self.__sampleInput: feed})
                feed = self.__sample(prediction)
                sentence += self.matrix2char(feed)
            self.echo(sentence)
        self.echo('=' * 80)


    '''
     计算真实 label 在 prediction 中的 log-probability
      cross_entropy = - sum( label * log(predictions) )
      log_prob = cross_entropy / N
    '''
    @staticmethod
    def __logProb(predictions, labels):
        predictions[predictions < 1e-10] = 1e-10    # 设置 predictions 中概率的最小值为 1e-10
        return - np.sum( np.multiply(labels, np.log(predictions)) ) / labels.shape[0]


    '''
     在一个分布中随机地采样一个样本; 返回值为样本在 distribution 中的 index
      采样的得到样本的概率 是符合 样本的分布
     :param
      [array] distribution ：一个 normalized 后的概率的数组
    '''
    @staticmethod
    def __sampleDistribution(distribution):
        r = random.uniform(0, 1)        # 因为 distribution 是符合随机分布，r 也取随机分布的值
        s = 0
        for i in range(len(distribution)):
            s += distribution[i]
            if s >= r:
                return i                # 返回在分布中的 index 位置
        return len(distribution) - 1    # 若 r 比分布中的所有值都大，则返回最后一个 index


    ''' 将 prediction 转为 one_hot_encoding '''
    @staticmethod
    def __sample(prediction):
        p = np.zeros(shape=[1, LSTM.VOCABULARY_SIZE], dtype=np.float)
        p[0, LSTM.__sampleDistribution(prediction[0])] = 1.0
        return p


    ''' 返回一个 shape 为 [1, vocabulary_size] 的随机分布 '''
    @staticmethod
    def __randomDistribution():
        b = np.random.uniform(0.0, 1.0, size=[1, LSTM.VOCABULARY_SIZE])
        return b / np.sum(b, 1)[:, None]


    ''' 矩阵转换为 string '''
    @staticmethod
    def matrix2char(matrix):
        return load.Download.id2Char(np.argmax(matrix, 1))


    def run(self):
        # 生成模型
        self.model()

        # 前向推导，用于预测准确率 以及 TensorBoard 里能同时看到 训练集、校验集 的准确率
        self.inference()

        # 正则化
        # self.__loss = self.regularize(self.__loss, self.REGULAR_BETA)

        # 生成训练的 op
        train_op = self.getTrainOp(self.__loss, self.__learningRate, self.globalStep)

        # 记录 loss 与 perplexity 的均值到 tensorboard
        self.__summaryMean()

        # 预测 (随机数据 或 校验集)
        self.__predict()

        # 初始化所有变量
        self.initVariables()

        # TensorBoard merge summary
        self.mergeSummary()

        best_perplexity_val = 1000              # 校验集 perplexity 最好的情况
        increase_perplexity_val_times = 0       # 校验集 perplexity 连续上升次数

        print '\nepoch\taccuracy_val:'

        mean_loss = 0
        mean_perplexity_train = 0

        for step in range(self.__steps):
            # 给 feed_dict 赋值
            batches = self.__trainSet.nextBatches(self.BATCH_SIZE, self.NUM_UNROLLINGS)
            feed_dict = {}
            for i, batch in enumerate(batches):
                feed_dict[self.__trainData[i]] = batch

            _, loss, perplexity_train = self.sess.run([train_op, self.__loss, self.__perplexity],
                                                feed_dict=feed_dict)      # 运行 训练

            mean_loss += loss
            mean_perplexity_train += perplexity_train

            if step != 0 and step % self.__summaryFrequency == 0:
                # 输出进度
                tmp_step = step / self.__summaryFrequency
                progress = 1.0 * step / self.__steps * 100.0
                self.echo('step: %d (%d) | %.2f%%           \r' % (step, self.__steps, progress))

                # 计算 loss、perplexity_train 的均值
                feed_dict[self.__lossPH] = mean_loss / float(self.__summaryFrequency)
                feed_dict[self.__perplexityPH] = mean_perplexity_train / float(self.__summaryFrequency)

                # # 计算 校验集的 perplexity
                # perplexity_val = self.__evaluate(self.__valSet, int(self.__valSize / 10000))
                # feed_dict[self.__perplexityVal] = perplexity_val

                self.addSummaryTrain(feed_dict, tmp_step)        # 输出数据到 TensorBoard

                # 计算 校验集的 perplexity
                perplexity_val = self.__evaluate(self.__valSet, int(self.__valSize / 10000))
                feed_dict[self.__perplexityPH] = perplexity_val

                self.addSummaryVal(feed_dict, tmp_step)

                # 重置 loss、perplexity_train 的均值
                mean_loss = 0
                mean_perplexity_train = 0

                # 随机预测文本，输出到 console 查看效果
                if step % (10 * self.__summaryFrequency) == 0:
                    self.__randomPredictText(2)

                # 判断 early stop 的条件
                if perplexity_val < best_perplexity_val:    # 若校验集的 peplexity 比 best_perplexity_val 低
                    best_perplexity_val = perplexity_val
                    increase_perplexity_val_times = 0

                    self.saveModel()                        # 保存模型

                else:                                       # 否则
                    increase_perplexity_val_times += 1
                    if increase_perplexity_val_times > self.EARLY_STOP_CONDITION:
                        break

        self.closeSummary() # 关闭 TensorBoard

        self.restoreModel() # 恢复模型

        # 计算 训练集、校验集、测试集 的 perplexity
        perplexity_train = self.__evaluate(self.__trainSet, self.__trainSize)
        perplexity_val = self.__evaluate(self.__valSet, self.__valSize)
        perplexity_test = self.__evaluate(self.__testSet, self.__testSize)

        print '\ntraining set perplexity: %.6f%%' % perplexity_train
        print 'validation set perplexity: %.6f%%' % perplexity_val
        print 'test set perplexity: %.6f%%' % perplexity_test

        self.echo('done')


o_nn = LSTM()
o_nn.run()

