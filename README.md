# belarusian-word-embedding
Code used to train word embeddings for belarusian words

## Files

`belarus-word-embeddings.ipynb` - this is the main file where embeddings are trained

`pretrained-embeddings-fasttext.ipynb` - this is an attempt to use pretrained FastText vectors available here https://fasttext.cc/docs/en/crawl-vectors.html

## Usage

Conda is used for working env, use `conda-requirements.txt` file to have appropriate packages.

## Analysis

Script to count most frequent words which were filtered out due to mismatch with vocabulary

```
cat removed-words.txt | sed 's/ /\n/g' | sort | uniq -c | sort -rn > removed-words-count.txt
```
