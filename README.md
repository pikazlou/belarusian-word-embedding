# belarusian-word-embedding
Code used to train word embeddings for belarusian words

## Files
`preprocess-text.ipynb` - run this notebook to fetch and process belarusian corpus from Common Crawl

`belarus-word-embeddings.ipynb` - this is the main file where embeddings are trained based on result of `preprocess-text.ipynb`

`pretrained-embeddings-fasttext.ipynb` - this is an attempt to use pretrained FastText vectors available here https://fasttext.cc/docs/en/crawl-vectors.html

## Usage

Conda is used for working env, use `conda-requirements.txt` file to have appropriate packages.

To use default set up just follow next steps, otherwise customize the code behind each step. Note, running steps 1 and 2 in default set up will take a lot of time!

1. Run `preprocessing.py` to download corpus and grammar and preprocess sentences
2. Run `train.py` to train different kind of models (comment out those you don't want)
3. Run `evaluation.ipynb` in notebook to evaluate trained models

## Analysis

Script to count most frequent words which were filtered out due to mismatch with vocabulary

```
cat removed-words.txt | sed 's/ /\n/g' | sort | uniq -c | sort -rn > removed-words-count.txt
```

## Related materials

https://arxiv.org/abs/1901.09785
https://github.com/vecto-ai/word-benchmarks
http://alfonseca.org/eng/research/wordsim353.html
https://code.google.com/archive/p/word2vec/
https://arxiv.org/pdf/1301.3781.pdf
https://aclanthology.org/W16-2507.pdf
https://github.com/mfaruqui/eval-word-vectors/blob/master/data/word-sim/EN-WS-353-REL.txt
https://marcobaroni.org/strudel/
http://lcl.uniroma1.it/outlier-detection/
https://nlp.stanford.edu/projects/glove/
https://aclanthology.org/D17-1024/
https://aclanthology.org/D14-1167.pdf
