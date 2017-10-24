#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import embedding
import load
import tensorflow as tf


'''
  Skip-gram 模型
'''
class SkipGram(embedding.Embedding):
    MODEL_NAME = 'skip_gram'    # 模型的名称

    # 模型存储的路径，默认使用 cbow 模型
    SAVE_PATH = r'../data/text8_bi_embedding_skip.pickle'


    ''' 初始化 X 与 y 的接口 （ 被 self.init() 调用 ） '''
    def initXY(self):
        # 输入 与 label
        self.__X = tf.placeholder(tf.int32, [self.BATCH_SIZE])  # 训练集数据
        self.__y = tf.placeholder(tf.int32, [self.BATCH_SIZE, 1])  # 训练集 label


    ''' 模型 '''
    def model(self):
        with tf.name_scope('layer_1'):
            self.__embeddings = tf.Variable(
                tf.random_uniform(self.SHAPE, -1.0, 1.0), name='embeddings'
            )

            self.__embed = tf.nn.embedding_lookup(self.__embeddings, self.__X, name='embed')

        with tf.name_scope('layer_2'):
            self.__W = self.initWeight(self.SHAPE)
            self.__b = self.initBias([self.VOCABULARY_SIZE])

    
    ''' 主函数 '''
    def run(self):
        # 生成模型
        self.model()

        # 计算 loss
        self.getLoss(self.__W, self.__b, self.__embed, self.__y)

        # 生成训练的 op
        train_op = self.getTrainOp(self.loss, self.learningRate, self.globalStep)

        # 计算相似度
        self.calSimilarity(self.__embeddings)

        # 初始化所有变量
        self.initVariables()

        # TensorBoard merge summary
        self.mergeSummary()

        best_loss = 999999                  # 记录训练集最好的 loss
        increase_loss_times = 0             # loss 连续上升的 epoch 次数

        for step in range(self.steps):

            if step % 50 == 0:                                  # 输出进度
                self.echo('step: %d (%d|%.2f%%) / %d|%.2f%%     \r' % (step, self.iterPerEpoch, 1.0 * step % self.iterPerEpoch / self.iterPerEpoch * 100.0, self.steps, 1.0 * step / self.steps * 100.0), False)

            batch_x, batch_y = self.data.skipGramNextBatch(self.BATCH_SIZE)
            feed_dict = {self.__X: batch_x, self.__y: batch_y}

            self.sess.run(train_op, feed_dict)                  # 运行 训练

            if step % self.iterPerEpoch == 0 and step != 0:   # 完成一个 epoch 时
                epoch = step // self.iterPerEpoch             # 获取这次 epoch 的 index
                loss = self.sess.run(self.loss, feed_dict)

                self.echo('\n****************** %d *********************' % epoch)

                self.addSummary(feed_dict, epoch)               # 输出数据到 TensorBoard

                # 用于输出与验证集对比的数据，查看训练效果
                sim = self.sess.run(self.similarity)
                for i in range(self.VALID_SIZE):
                    valid_word = load.TransFormBi.id2Bi(self.valExample[i])

                    top_k = 8                                           # 设置最近的邻居个数
                    nearest = (-sim[i, :]).argsort()[1 : top_k + 1]     # 获取最近的 top_k 个邻居

                    # 输出展示最近邻居情况
                    log = 'Nearest to %s: ' % valid_word
                    for k in range(top_k):
                        log += load.TransFormBi.id2Bi(nearest[k]) + ' '
                    self.echo(log)

                if loss < best_loss:                    # 当前 loss 比 最好的 loss 低，则更新 best_loss
                    best_loss = loss
                    increase_loss_times = 0

                    self.saveModel()                    # 保存模型

                else:                                   # 否则
                    increase_loss_times += 1
                    if increase_loss_times > 30:
                        break

        self.closeSummary()  # 关闭 TensorBoard

        self.restoreModel()  # 恢复模型

        # 获取最终需要的词向量
        final_embedding = self.sess.run(self.normalizeEmbedding)
        # 保存 embeddings 到文件
        self.saveEmbedding(final_embedding)
        # 把 词向量 降维后画成图像
        self.plot2dEmbeddingSim(final_embedding, 400)


o_nn = SkipGram()
o_nn.run()
