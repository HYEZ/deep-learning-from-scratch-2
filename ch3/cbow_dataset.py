import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_contexts_target, convert_one_hot

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(id_to_word)

'''
직접 구현
'''
# target = np.array([0 for i in range(len(corpus) - 2)])
# contexts = np.array([[0, 0] for i in range(len(corpus) - 2)])
# print(target)
# for i in range(1, len(corpus)-1):
# 	target[i-1] = corpus[i]
# 	print(contexts[i-1])
# 	contexts[i-1] = np.array([corpus[i-1], corpus[i+1]])

contexts, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(target, vocab_size)

print('context', contexts)
print('target', target)