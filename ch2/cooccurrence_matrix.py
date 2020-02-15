import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar

text = 'You say goodbye I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['say']]
c1 = C[word_to_id['goodbye']]
print(cos_similarity(c0, c1))

most_similar('hello', word_to_id, id_to_word, C, top=5)