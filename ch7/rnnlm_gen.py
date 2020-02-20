import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch6.rnnlm import Rnnlm
from ch6.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
	def generate(self, start_id, skip_ids=None, sample_size=100):
		word_ids = [start_id]

		x = start_id
		while len(word_ids) < sample_size:
			x = np.array(x).reshape(1, 1) # predict() 함수는 미니배치 처리를 위해 2차원이라서 2차원으로 만들어줌
			score = self.predict(x)
			p = softmax(score.flatten())

			sampled = np.random.choice(len(p), size=1, p=p)
			if(skip_ids is None) or (sampled not in skip_ids):
				x = sampled
				word_ids.append(int(x))

		return word_ids

class BetterRnnlmGen(BetterRnnlm):
	def generate(self, start_id, skip_ids=None, sample_size=100):
		word_ids = [start_id]

		x = start_id
		while len(word_ids) < sample_size:
			x = np.array(x).reshape(1, 1) # predict() 함수는 미니배치 처리를 위해 2차원이라서 2차원으로 만들어줌
			score = self.predict(x)
			p = softmax(score.flatten())

			sampled = np.random.choice(len(p), size=1, p=p)
			if(skip_ids is None) or (sampled not in skip_ids):
				x = sampled
				word_ids.append(int(x))

		return word_ids