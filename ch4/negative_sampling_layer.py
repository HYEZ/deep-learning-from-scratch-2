# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h # 벡터의 내적은 곱셈노드 + 덧셈노드 => 미분하면 하면 h, target_W 바뀜
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler: # 한 단어를 대상으로 확률분포를 만든다.
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None # 확률 분포

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target): # target 인수로 지정한 단어를 positive로 해석, 그 외의 단어 id를 샘플링
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32) # 샘플링 배열(결과)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0 # 정답데이터는 샘플링 안되게 함
                p /= p.sum() # 확률로 다시 만들어줌 (총합 1되게)
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)] # 정답있는 레이어 포함해서 (샘플링 + 1)
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        # 매개변수, 기울기 한 곳에 넣음
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 긍정적인 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32) # 정답 레이블 = 1
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적인 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32) # negative 레이블 = 0
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i] # i번째 열벡터 가져오기
            score = self.embed_dot_layers[1 + i].forward(h, negative_target) # h는 모든 계층에 같은 값 줌
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore) # repeat node라서 다 더함

        return dh
