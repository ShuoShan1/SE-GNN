import keras
import numpy as np
from utils import *

import tensorflow as tf
import keras.backend as K
from keras.layers import *


class evaluate:
    def __init__(self, dev_pair):
        self.dev_pair = dev_pair

        Matrix_A = Input(shape=(None, None))
        Matrix_B = Input(shape=(None, None))

        def dot(tensor):
            k = 10
            A, B = [K.squeeze(matrix, axis=0) for matrix in tensor]
            A_sim = K.dot(A, K.transpose(B))
            return K.expand_dims(A_sim, axis=0)

        results = Lambda(dot)([Matrix_A, Matrix_B])
        self.sim_model = keras.Model(inputs=[Matrix_A, Matrix_B], outputs=results)

        k = 10
        Matrix_A = Input(shape=(None, None))
        results = Lambda(
            lambda x: K.expand_dims(K.sum(tf.nn.top_k(K.squeeze(x, axis=0), k=k)[0], axis=-1) / k, axis=0))(Matrix_A)
        self.avg_model = keras.Model(inputs=[Matrix_A], outputs=results)

        Matrix_A = Input(shape=(None, None))
        LR_input = Input(shape=(None,))
        RL_input = Input(shape=(None,))
        Ans_input = Input(shape=(None,))

        def CSLS(tensor):#[R_sim[epoch], np.concatenate(LR, axis=1), RL[epoch], np.expand_dims(ans_rank, axis=0)])
            sim, LR, RL, ans_rank = [K.squeeze(m, axis=0) for m in tensor]
            LR, RL, ans_rank = [K.expand_dims(m, axis=1) for m in [LR, RL, ans_rank]]
            sim = 2 * sim - K.transpose(LR)
            sim = sim - RL
            rank = tf.argsort(-sim, axis=-1)#argsort从小到大排序 但是-sim 就是从大到小，返回的是按行的索引矩阵
            results = tf.where(tf.equal(rank, K.cast(K.tile(ans_rank, [1, len(self.dev_pair)]), dtype="int32")))
            return K.expand_dims(results, axis=0)

        results = Lambda(CSLS)([Matrix_A, LR_input, RL_input, Ans_input])
        self.CSLS_model = keras.Model(inputs=[Matrix_A, LR_input, RL_input, Ans_input], outputs=results)

    def CSLS_cal(self, Lvec, Rvec, evaluate=True, batch_size=1024):
        L_sim, R_sim = [], []
        for epoch in range(len(Lvec) // batch_size + 1):
            L_sim.append(self.sim_model.predict(
                [np.expand_dims(Lvec[epoch * batch_size:(epoch + 1) * batch_size], axis=0),
                 np.expand_dims(Rvec, axis=0)]))
            R_sim.append(self.sim_model.predict(
                [np.expand_dims(Rvec[epoch * batch_size:(epoch + 1) * batch_size], axis=0),
                 np.expand_dims(Lvec, axis=0)]))

        LR, RL = [], []
        for epoch in range(len(Lvec) // batch_size + 1):
            LR.append(self.avg_model.predict([L_sim[epoch]]))#选出前K个余弦相似度最高的值,再平均
            RL.append(self.avg_model.predict([R_sim[epoch]]))

        if evaluate:
            results = []
            for epoch in range(len(Lvec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size, min((epoch + 1) * batch_size, len(Lvec)))])
                result = self.CSLS_model.predict(
                    [R_sim[epoch], np.concatenate(LR, axis=1), RL[epoch], np.expand_dims(ans_rank, axis=0)])
                results.append(result)
            return np.concatenate(results, axis=1)[0]
        else:
            l_rank, r_rank = [], []
            for epoch in range(len(Lvec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size, min((epoch + 1) * batch_size, len(Lvec)))])
                r_rank.append(self.rank_model.predict(
                    [R_sim[epoch], np.concatenate(LR, axis=1), RL[epoch], np.expand_dims(ans_rank, axis=0)]))
                l_rank.append(self.rank_model.predict(
                    [L_sim[epoch], np.concatenate(RL, axis=1), LR[epoch], np.expand_dims(ans_rank, axis=0)]))

            return np.concatenate(r_rank, axis=1)[0], np.concatenate(l_rank, axis=1)[0]
