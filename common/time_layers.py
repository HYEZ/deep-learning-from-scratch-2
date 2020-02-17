import numpy as np

class RNN:
	def __init__(self, Wx, Wh, Wb): # 입력 가중치, 다음 시각 출력으로 변환을 위한 가중치, 편향
		self.params = [Wx, Wh, Wb]
		self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wb)]
		self.cache = None

	def forward(self, x, h_prev):
		Wx, Wh, b = self.params
		t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
		h_next = np.tanh(t)

		self.cache = (x, h_prev, h_next)
		return h_next


	def backward(self, dh_next):
		Wx, Wh, b = self.params
		x, h_prev, h_next = self.cache

		dt = dh_next * (1 - h_next ** 2) # tanh
		db = np.sum(dt, axis=0) # bias (repeat 노드라서)
		dWh = np.matmul(h_prev.T, dt)
		dh_prev = np.matmul(dt, Wh.T)
		dWx = np.matmul(x.T, dt)
		dx = np.matmul(dt, Wx.T)

		self.grads[0][...] = dWx
		self.grads[1][...] = dWh
		self.grads[2][...] = db

		return dx, dh_prev

