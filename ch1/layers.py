import numpy as np

# 시그모이드 계층
class Sigmoid:
	def __init__(self):
		self.params = [] # 시그모이드 계층은 학습해야할 매개변수가 없으므로 빈 리스트로 초기화
		self.grads = []
		self.out = None

	def forward(self, x):
		out = 1 / (1 + np.exp(-x))
		self.out = out
		return out

	def backward(self, dout):
		dx = dout * self.out * (1.0 - self.out)
		return dx

# Affine 계층
class Affine:
	def __init__(self, W, b):
		self.params = [W, b]
		self.grads = [np.zeros_like(W), np.zeros_like(b)]
		self.x = None

	def forward(self, x):
		W, b = self.params
		out = np.matmul(x, W) + b
		self.x = x
		return out

	def backward(self, dout):
		W, b = self.params
		dx = np.matmul(dout, W.T)
		dW = np.matmul(self.x.T, dout)
		db = np.sum(dout, axis=0)

		self.grads[0][...] = dW
		self.grads[1][...] = db
		return dx

# 행렬 곱셈 계층
class MatMul:
	def __init__(self, W):
		self.params = [W]
		self.grads = [np.zeros_like(W)]
		self.x = None

	def forward(self, x):
		W, = self.params
		out = np.matmul(x, W)
		self.x = x
		return out

	def backward(self, dout):
		W, = self.params
		dx = np.matmul(dout, W.T)
		dW = np.matmul(self.x.T, dout)
		self.grads[0][...] = dW
		return dx

