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

class TimeRNN:
	def __init__(self, Wx, Wh, b, stateful=False):
		self.params = [Wx, Wh, b]
		self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
		self.layers = None # 다수의 RNN 계층을 리스트로 저장

		self.h, self.dh = None, None # h: forward()시 마지막 RNN 계층의 은닉 상태(h_next) / dh: backward()시 하나 앞 블록의 은닉 상태의 기울기(dh_prev)
		self.stateful = stateful # True: h 유지(순전파 끊기지 않음), False: 은닉상태 h를 영행렬로 초기화함

	def set_state(self, h):
		self.h = h

	def reset_state(self):
		self.h = None

	def forward():
		Wx, Wh, b = self.params
		N, T, D = xs.shape
		D, H = Wx.shape

		self.layers = []
		hs = np.empty((N, T, H), dtype='f')

		if not self.stateful or self.h is None: # stateful이 False or 처음 호출 시
			self.h = np.zeros((N, H), dtype='f') # 0행렬로 초기화

		for t in range(T):
			layer = RNN(*self.params) # *는 리스트의 원소들을 추출하여 메서드의 인수로 전달함. 
			self.h = layer.forward(xs[:, t, :], self.h)
			hs[:, t, :] = self.h
			self.layers.append(layer)

		return hs

	def backward(self, dhs):
		Wx, Wh, b = self.params
		N, T, H = dhs.shape
		D, H = Wx.shape

		dxs = np.empty((N, T, D), dtype='f')
		dh = 0
		grads = [0, 0, 0] # Wx, Wh, b
		for t in reversed(range(T))
			layer = self.layers[t]
			dx, dh = layer.backward(dhs[:, t, :] + dh) # 합산된 기울기
			dxs[:, t, :] = dx

			for i, grad in enumerate(layer.grads):
				# RNN 계층들은 같은 가중치를 사용하므로
				# TimeRNN의 최종 가중치의 기울기는 각 RNN 계층의 가중치 기울기를 다 더한 것이 됨
				grads[i] += grad 

		for i, grad in enumerate(grads):
			self.grads[i][...] = grad
		self.dh = dh

		return dxs
