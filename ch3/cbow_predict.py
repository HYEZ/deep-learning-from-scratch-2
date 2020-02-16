import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul

# 샘플 맥락 데이터
c0 = np.array([[1,0,0,0,0,0,0]])
c1 = np.array([[0,0,1,0,0,0,0]])

# 가중치 초기화
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 계층 생성
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 순전파
h0 = in_layer0.forward(c0)
h1 = in_layer0.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)
print(s)

# ###### input layer 하나로 만들어봄 #####
# c = np.array([[1,0,0,0,0,0,0], [0,0,1,0,0,0,0]])
# in_layer = MatMul(W_in)
# out_layer2 = MatMul(W_out)

# h2 = in_layer0.forward(c)
# h2 = h2.sum(axis=0)/2
# s2 = out_layer2.forward(h)
# print(s2)

