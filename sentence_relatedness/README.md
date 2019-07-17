<h3>Calculating Sentence relatedness</h1>

**Current Approach**
* Extract candidate phrases: 
      ```{(<JJ.*>*<NN.*>+<IN>)?<JJ>*<NN.*>+}, 
	{<NN.*>+<VB.*>+}, 
	{<VB.*>+<NN.*>+}```
* Generate noun-phrase embedding - sum of corresponding BERT token features
* Calculate cosine similarity between noun-phrases of sentences
* Final score is the maximum of all pair-wise scores
* Walkthrough notebook: `playground.ipynb`
* Sample run: `similarity.py`

