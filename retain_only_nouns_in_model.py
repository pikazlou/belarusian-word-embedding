import sys
import json
import numpy as np
from gensim.models import KeyedVectors


def restrict_w2v(w2v, restricted_word_set):
    new_index_to_key = []
    new_key_to_index = {}
    new_vectors = []
    for ind, word in enumerate(w2v.index_to_key):
        if word in restricted_word_set:
            new_key_to_index[word] = len(new_index_to_key)
            new_index_to_key.append(word)
            new_vectors.append(w2v.vectors[ind])
    w2v.index_to_key = new_index_to_key
    w2v.key_to_index = new_key_to_index
    w2v.vectors = np.array(new_vectors)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "2 parameters expected: <input_keyed_vectors_file> <output_keyed_vectors_file>"
    input_vectors_file = sys.argv[1]
    output_vectors_file = sys.argv[2]

    with open('word-map-only-nouns.json') as f:
        word_map = json.load(f)

    allowed_words = set(word_map.values())
    vectors = KeyedVectors.load_word2vec_format(input_vectors_file)
    restrict_w2v(vectors, allowed_words)
    vectors.save_word2vec_format(output_vectors_file)





