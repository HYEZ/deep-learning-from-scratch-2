import numpy as np

class EmbeddingDot:
	def __init__(self, W):
		self.emebed = Embedding(W)
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
		self.emebed.backward(dtarget_W)
		dh = dout * target_W
		return dh