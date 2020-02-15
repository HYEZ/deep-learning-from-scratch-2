import numpy as np

# 시그모이드 계층
class Sigmoid:
	def __init__(self):
		self.params = [] # 시그모이드 계층은 학습해야할 매개변수가 없으므로 빈 리스트로 초기화

	def forward(self, x):
		return 1 / (1 + np.exp(-x))

# Affine 계층
class Affine:
	def __init__(self, W, b):
		self.params = [W, b]

	def forward(self, x):
		W, b = self.params
		out = np.matmul(x, W) + b
		return out


# 신경망 구현
class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size):
		I, H, O = input_size, hidden_size, output_size

		# 가중치와 편향 초기화
		W1 = np.random.randn(I, H)
		b1 = np.random.randn(H)
		W2 = np.random.randn(H, O)
		b2 = np.random.randn(O)

		# 계층 생성
		self.layers = [
			Affine(W1, b1),
			Sigmoid(),
			Affine(W2, b2)
		]


		# 모든 가중치를 리스트에 모음 
		# 매개변수를 하나의 리스트에 보관하면 배개변수 갱신과 저장을 쉽게할 수 있음
		self.params = []
		for layer in self.layers:
			self.params += layer.params
			print("1", layer.params)
		print(self.params)

	def predict(self, x):
		for layer in self.layers:
			x = layer.forward(x) # x 갱신해서 차례로 forward 수행
		return x


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
# print(s)