import numpy as np
from gensim.models import KeyedVectors, Word2Vec, FastText


if __name__ == "__main__":
    model = Word2Vec.load("w2v-sg0-d100-w3-min100-nse0-smp10000-ep500-descr_xsteeper_alpha.model")
    word = model.wv.index_to_key[np.random.randint(5000)]
    ranked_list = [pair[0] for pair in model.wv.most_similar(word, topn=len(model.wv.index_to_key))]
    ranked_list = [word] + ranked_list
    best_rank = 999999
    while best_rank != 0:
        s = input()
        try:
            ind = ranked_list.index(s)
            print(ind)
            if ind < best_rank:
                best_rank = ind
        except ValueError:
            pass

    print('CONGRATS!')
    for i, w in enumerate(ranked_list[:20]):
        print(f'{i}\t{w}')



