import sys
sys.path.append('..')
import numpy as np
from common.layers import *

# 신경망 구현
class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size):
		I, H, O = input_size, hidden_size, output_size

		# 가중치와 편향 초기화
		W1 = 0.01 * np.random.randn(I, H)
		b1 = np.zeros(H)
		W2 = 0.01 * np.random.randn(H, O)
		b2 = np.zeros(0)

		# 계층 생성
		self.layers = [
			Affine(W1, b1),
			Sigmoid(),
			Affine(W2, b2)
		]
		self.loss_layer = SoftmaxWithLoss()

		# 모든 가중치를 리스트에 모음 
		# 이 모델에서 사용하는 매개변수들과 기울기를 각각 하나로 모음
		# 매개변수를 하나의 리스트에 보관하면 매개변수 갱신과 저장을 쉽게할 수 있음
		self.params, self.grads = [], []
		for layer in self.layers:
			self.params += layer.params
			self.grads += layer.grads 
		print(self.params)


	def predict(self, x):
		for layer in self.layers:
			x = layer.forward(x) # x 갱신해서 차례로 forward 수행
		return x

	def forward(self, x, t):
		score = self.predict(x)
		loss = self.loss_layer.forward(score, t)
		return loss

	def backward(self, dout=1):
		dout = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout



# x = np.random.randn(10, 2)
# model = TwoLayerNet(2, 4, 3)
# s = model.predict(x)
# print(s)