# Turkish-Multiple-Choice-Question-Answering

In this repository, I combine Turkish multiple-choice exam dataset and show baseline bert results.
You can find an example of multiple choice training ipynb file and dataset folder contain train and dev files.
Dataset contains a set of items which contains question, choices, related paragraph and related subject.


### DATASET
|    Dataset    |  Five Choices  |    Four Choices      |          Total               |
| ------------- |:--------------:|:--------------------:|:----------------------------:|
|    Train      |     1136       |       2159           |          3295                |
|     Dev       |     137        |       411            |          548                 |


### MODEL
|           Name             | epoch | max_seq_length | learning_rate | per_gpu_train_batch_size |
|:---------------------------|:-----:|:--------------:|:-------------:|:------------------------:|
|   Bert-Base-Turkish-Cased  |   3   |      256       |     5e-5      |            8             |


### RESULTS
|           Name             | epoch |    eval_acc    |   eval_loss   |
|:---------------------------|:-----:|:--------------:|:-------------:|
|   Bert-Base-Turkish-Cased  |   3   |    0.77696     |    0.73243    |         

You can access the model from https://huggingface.co/enelpi/bert-turkish-multiple-choice-fine-tuned/tree/main here.


### Requirements
transformers==2.8.0
tqdm==4.50.0
tokenizers==0.5.2       
sentencepiece==0.1.91  



You can find more detailed description and source code from https://github.com/mhardalov/exams-qa here. This repository contains different multiple-choice question answering datasets and I found Turkish one here.
Also here is the related paper.
M. Hardalov, T. Mihaylov, D. Zlatkova, Y. Dinkov, I. Koychev, P. Nakov [*EXAMS: A Multi-subject High School Examinations Dataset for
Cross-lingual and Multilingual Question Answering*](http://arxiv.org/abs/2011.03080)
