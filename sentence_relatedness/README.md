#Calculating Sentence relatedness bwtewwn two sentences
*Calculates cosine similarity between nown phrases in the sentences
**Currnet Approach
* Extract candidate noun-phrases - `{(<JJ.*>*<NN.*>+<IN>)?<JJ>*<NN.*>+}`
* Generate noun-phrase embedding - sum of corresponding BERT token features
* Calculate cosine similarity between noun-phrases of sentences
* Final score is the maximum of all pair-wise scores

Walkthrough notebook: `playground.ipynb`
Sample run: `similarity.py`
