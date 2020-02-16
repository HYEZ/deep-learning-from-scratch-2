import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import *

text = 'You say goodbye I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

#SVD
U, S, V = np.linalg.svd(W)
print(W.shape, U.shape, S.shape, V.shape)

print('C', C[0]) # 동시발생행렬
print('W', W[0]) # ppmi
print('U', U[0]) # SVD
print('S', S) 
print('V', V[0])
print(U[0, :2]) # 차원 감소
K = U[:,:3]
print(K.shape)
for word, word_id in word_to_id.items():
	plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
# plt.show()