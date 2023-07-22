The dataset and corresponding validation (outlier detection) is based on 8-8-8 dataset for intrinsic evaluation of word embedding.
Reference implementation was taken from here: http://lcl.uniroma1.it/outlier-detection/

In short, the idea is to have 8 related words ("cluster") for a particular topic together with 8 other words ("outliers") that might be also relevant but not as close as first 8 words.
During evaluation, the algorithm takes cluster words from a single file and adds one of outliers from that file.
The model then evaluates similarity between each pair of words. The word with the lowest sum of similarities is considered to be outlier.
If outlier predicted by model is equal to actual outlier, the model scores 1. Otherwise, the model scores 0.
The average of scores over all possible evaluation sets is the final score for the model.

Additional good overview of word embedding evaluation is available in "Evaluating Word Embedding Models: Methods and Experimental Results": https://arxiv.org/abs/1901.09785
