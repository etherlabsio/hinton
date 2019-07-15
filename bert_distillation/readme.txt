models.py --- has child network class in the end..has parent bert functions in the beginning of the code
data.py ---- loading ether sw data and contains classes for data preprocessing
distillation.py ---train and eval the distillation 
classiefier.py ---- train/eval the nsp context woindow classifier
distill_utils.py --- extract nsp scores between two sentences and extract feature vector for a sentence
EtherText.json --- location for ether data and vocab.txt
optim.json --- all hyperparameters should be changed here for training of classifier.py/distillation.py

References: https://arxiv.org/pdf/1903.12136.pdf
https://openreview.net/pdf?id=HJxM3hftiX
https://arxiv.org/pdf/1907.02226.pdf
https://github.com/peterliht/knowledge-distillation-pytorch
https://github.com/shudima/notebooks/blob/master/Distilling_Bert.ipynb
https://arxiv.org/pdf/1708.06128.pdf
