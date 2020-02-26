import sys
sys.path.append('..')
from common.time_layers import *
from ch7.seq2seq import Encoder, Seq2seq
from ch8.attention_layer import TimeAttention

class AttentionEncoder(Encoder):
	def forward(self, xs):
		xs = self.embed.forward(xs)
		hs = self.lstm.forward(xs)
		return hs

	def backward(self, dhs):
		dout = self.lstm.backward(dhs)
		dout = self.embed.backward(dout)
		return dout

