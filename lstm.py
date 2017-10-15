#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

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
    MODEL_NAME = 'lstm'                             # 模型的名称

    BASE_LEARNING_RATE = 10.0                       # 初始 学习率
    DECAY_RATE = 0.1                                # 学习率 的 下降速率

    CLIP_NORM = 1.25                                # 梯度剪切的参数

    NUM_UNROLLINGS = 10                             # 序列数据一次输入 nul_unrollings 个数据
    BATCH_SIZE = 64                                 # 随机梯度下降的 batch 大小
    EPOCH_TIMES = 200                               # 迭代的 epoch 次数

    VOCABULARY_SIZE = load.Download.VOCABULARY_SIZE # 词典大小
    NUM_NODES = 64                                  # layer1 有多少个节点

    SHAPE_DATA = [BATCH_SIZE, VOCABULARY_SIZE]      # 输入 data 的 shape
    SHAPE_INPUT = [VOCABULARY_SIZE, NUM_NODES]      # cell 中与输入相乘的权重矩阵的 shape
    SHAPE_OUTPUT = [NUM_NODES, NUM_NODES]           # cell 中与上次输出相乘的权重矩阵的 shape

    # REGULAR_BETA = 0.01                             # 正则化的 beta 参数
    # MOMENTUM = 0.9                                  # 动量的大小


    ''' 自定义 初始化变量 过程 '''
    def init(self):
        # 加载数据
        self.load()

        # 常量
        self.__iterPerEpoch = int(self.__trainSize // self.BATCH_SIZE)
        self.__steps = self.EPOCH_TIMES * self.__iterPerEpoch

        # TODO 删除该提示
        print 'iter_per_epoch: %d' % self.__iterPerEpoch
        print 'steps: %d' % self.__steps

        # 输入 与 label
        self.__trainData = list()
        for _ in range(self.NUM_UNROLLINGS + 1):
            self.__trainData.append( tf.placeholder(tf.float32, shape=self.SHAPE_DATA, name='input') )
        self.__X = self.__trainData[:self.NUM_UNROLLINGS]
        self.__y = self.__trainData[1:]                       # label 为 输入数据往后移一位 (序列数据)

        # 随训练次数增多而衰减的学习率
        self.__learningRate = self.getLearningRate(
            self.BASE_LEARNING_RATE, self.globalStep, self.BATCH_SIZE, self.__steps, self.DECAY_RATE
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
        # self.__valSet = load.Data(0.64, 0.8)            # 按 0.16 的比例划分校验集
        # self.__testSet = load.Data(0.8)                 # 按 0.2  的比例划分测试集

        self.__trainSize = self.__trainSet.getSize()
        # self.__valSize = self.__valSet.getSize()
        # self.__testSize = self.__testSet.getSize()


    ''' 模型 '''
    def model(self):
        # ********************** 初始化模型所需的变量 **********************

        with tf.name_scope('cell'):

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
            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.__logits, labels=tf.concat(self.__y, 0))
            )


    ''' 获取预测值 '''
    def inference(self):
        with tf.name_scope('predict'):
            # lstm 的输出
            self.__predict = tf.nn.softmax(self.__logits, name='predict')

        with tf.name_scope('perplexity'):
            # 计算 perplexity
            labels = tf.concat(self.__y, 0, name='labels')
            self.__perplexity = tf.exp(
                - tf.reduce_sum( tf.log(self.__predict, name='log_predict') * labels ) /
                tf.cast(tf.shape(labels)[0], dtype=tf.float32), name='perplexity')

        # 将 perplexity 记录到 summary 中
        tf.summary.scalar('perplexity', self.__perplexity)


    ''' 获取 train_op '''
    def getTrainOp(self, loss, learning_rate, global_step):
        tf.summary.scalar('loss', loss)     # TensorBoard 记录 loss

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.CLIP_NORM)
            return optimizer.apply_gradients(zip(gradients, v), global_step=global_step)


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


    ''' 初始化 校验集 或 测试集 预测所需的变量 '''
    def __predict(self):
        self.__sampleInput = tf.placeholder(tf.float32, shape=[1, self.VOCABULARY_SIZE],
                                            name='sample_input')
        self.__sampleSavedOutput = tf.Variable( tf.zeros([1, self.NUM_NODES]), name='sample_saved_output' )
        self.__sampleSavedState = tf.Variable( tf.zeros([1, self.NUM_NODES]), name='sample_saved_state' )
        self.__sampleResetState = tf.group(
            self.__sampleSavedOutput.assign(tf.zeros([1, self.NUM_NODES])),
            self.__sampleSavedState.assign(tf.zeros([1, self.NUM_NODES]))
        )
        self.__sampleOutput, self.__sampleState = self.__cell(self.__sampleInput, self.__sampleSavedOutput, self.__sampleSavedState)

        with tf.control_dependencies([
            self.__sampleSavedOutput.assign(self.__sampleOutput),
            self.__sampleSavedState.assign(self.__sampleState)
        ]):
            self.__samplePrediction = tf.nn.softmax( tf.nn.xw_plus_b(self.__sampleOutput, self.__w, self.__b) )


    # ''' 计算准确率 '''
    # @staticmethod
    # def __getAccuracy(labels, predict, _size, name = ''):
    #     if name:
    #         scope_name = '%s_accuracy' % name
    #     else:
    #         scope_name = 'accuracy'
    #
    #     with tf.name_scope(scope_name):
    #         labels = tf.argmax(labels, 1)
    #         predict = tf.argmax(predict, 1)
    #         correct = tf.equal(labels, predict) # 返回 predict 与 labels 相匹配的结果
    #
    #         accuracy = tf.divide( tf.reduce_sum( tf.cast(correct, tf.float32) ), _size ) # 计算准确率
    #         if name: # 将 准确率 记录到 TensorBoard
    #             tf.summary.scalar('accuracy', accuracy)
    #
    #         return accuracy
    #
    #
    # ''' 使用不同数据 评估模型 '''
    # def evaluation(self, data_set, batch_size, accuracy = None):
    #     batch_x, batch_y = data_set.nextBatch(batch_size)
    #     return self.sess.run(accuracy, {
    #         self.__preX: batch_x, self.__preY: batch_y, self.__preSize: batch_x.shape[0]
    #     })


    def run(self):
        # 生成模型
        self.model()

        # 前向推导，用于预测准确率 以及 TensorBoard 里能同时看到 训练集、校验集 的准确率
        self.inference()

        # 正则化
        # self.__loss = self.regularize(self.__loss, self.REGULAR_BETA)

        # 生成训练的 op
        train_op = self.getTrainOp(self.__loss, self.__learningRate, self.globalStep)

        # ret_accuracy_train = self.__getAccuracy(self.__y, self.__output, self.__size, name='training')
        # ret_accuracy_val = self.__getAccuracy(self.__preY, self.__predict, self.__preSize, name='validation')

        # 初始化所有变量
        self.initVariables()

        # TensorBoard merge summary
        self.mergeSummary()

        # best_accuracy_val = 0           # 校验集准确率 最好的情况
        # decrease_acu_val_times = 0      # 校验集准确率连续下降次数
        #
        # # 获取校验集的数据，用于之后获取校验集准确率，不需每次循环重新获取
        # batch_val_x, batch_val_y = self.__valSet.nextBatch(self.__valSize)
        # batch_val_size = batch_val_x.shape[0]
        #
        print '\nepoch\taccuracy_val:'

        for step in range(self.__steps):
            if step % 50 == 0:                          # 输出进度
                epoch_progress = 1.0 * step % self.__iterPerEpoch / self.__iterPerEpoch * 100.0
                step_progress = 1.0 * step / self.__steps * 100.0
                self.echo('step: %d (%d|%.2f%%) / %d|%.2f%%     \r' % (step, self.__iterPerEpoch,
                                                    epoch_progress, self.__steps, step_progress), False)

            # 给 feed_dict 赋值
            batches = self.__trainSet.nextBatches(self.BATCH_SIZE, self.NUM_UNROLLINGS)
            feed_dict = {}
            for i, batch in enumerate(batches):
                feed_dict[self.__trainData[i]] = batch

            self.sess.run(train_op, feed_dict=feed_dict)      # 运行 训练
            # self.sess.run([train_op, self.__perplexity], feed_dict=feed_dict)      # 运行 训练

            if (step % self.__iterPerEpoch == 0 or step % 1000 == 0) and step != 0: # 完成一个 epoch 时
                self.sess.run(self.__perplexity, feed_dict=feed_dict)

                epoch = step // self.__iterPerEpoch     # 获取这次 epoch 的 index
                print '%d             ' % epoch

                # accuracy_val = self.evaluation(self.__valSet, self.__valSize, ret_accuracy_val) # 获取校验集准确率
                # print '\n%d\t%.10f%%' % (epoch, accuracy_val)

        #         feed_dict = {self.__X   : batch_x,      self.__y    : batch_y,      self.__size     : batch_x.shape[0],
        #                      self.__preX: batch_val_x,  self.__preY : batch_val_y,  self.__preSize  : batch_val_size}

                self.addSummary(feed_dict, int(step / 1000))       # 输出数据到 TensorBoard
                # self.addSummary(feed_dict, epoch)       # 输出数据到 TensorBoard
        #
        #         if accuracy_val > best_accuracy_val:    # 若校验集准确率 比 最高准确率高
        #             best_accuracy_val = accuracy_val
        #             decrease_acu_val_times = 0
        #
        #             self.saveModel()                    # 保存模型
        #
        #         else:                                   # 否则
        #             decrease_acu_val_times += 1
        #             if decrease_acu_val_times > 10:
        #                 break
        #
        self.closeSummary() # 关闭 TensorBoard

        self.restoreModel() # 恢复模型

        # # 计算 训练集、校验集、测试集 的准确率
        # accuracy_train = self.evaluation(self.__trainSet, self.__trainSize, ret_accuracy_val)
        # accuracy_val = self.evaluation(self.__valSet, self.__valSize, ret_accuracy_val)
        # accuracy_test = self.evaluation(self.__testSet, self.__testSize, ret_accuracy_val)
        #
        # print '\ntraining set accuracy: %.6f%%' % accuracy_train
        # print 'validation set accuracy: %.6f%%' % accuracy_val
        # print 'test set accuracy: %.6f%%' % accuracy_test


o_nn = LSTM()
o_nn.run()

