import numpy as np

class Embedding:
	def __init__(self, W):
		self.params = [W]
		self.grads = [np.zeros_like(W)]
		self.idx = None


	def forward(self, idx):
		# print(W.shape)
		W, = self.params
		self.idx = idx
		out = W[idx]
		return out

	def backward(self, dout):
		dW, = self.grads
		dW[...] = 0
		print(self.idx)
		# for i, word_id in enumerate(self.idx):
		# 	print(word_id, dout[i])
		# 	dW[word_id] += dout[i]
		# print(dW)
		np.add.at(dW, self.idx, dout)
		print(dW)
		return None


W = np.random.randn(7, 3)
layer = Embedding(W)
idx = [0, 1, 1]
a = layer.forward(idx)
b = layer.backward([[1, 2, 3], [2, 3, 4], [4, 5, 6]])
print(a, b)