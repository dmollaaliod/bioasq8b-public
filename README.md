# bioasq8b

## What is this repository for? ###

This code implements Macquarie University's experiments and
participation in BioASQ 8b.
* [BioASQ](http://www.bioasq.org)
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

## How do I get set up? ###

Apart from the code in this repository, you will need the following files:

* `BioASQ-training8b.json` - available from [BioASQ](http://www.bioasq.org/)
* `rouge_8b.csv` - you can create it by running the following overnight:
```
>>> from regression import saveRouge
>>> saveRouge('BioASQ-training8b.json', 'rouge_8b.csv',
               snippets_only = True)
```
* `allMeSH_2016_100.vectors.txt`, `allMeSH_2016_200.vectors.txt` - ask diego.molla-aliod@mq.edu.au.

* If you want to train the version that uses BioBERT, download the pre-trained weights 
from https://github.com/dmis-lab/biobert and convert them to the huggingface tensorflow
(Keras) interface. You can read instructions about how to convert them here: 
https://stackoverflow.com/questions/60539758/biobert-for-keras-version-of-huggingface-transformers
Alternatively, ask diego.molla-aliod@mq.edu.au for a copy of the BioBERT model.

Read the file `Dockerfile` for an idea of how to install the dependencies and
set up the system.

## Reading

* BioASQ8b paper (TBA)

## Examples of runs using pre-learnt models

To get the pre-learnt models used in the BioASQ8b runs, please email
diego.molla-aliod@mq.edu.au. The following models are available:
    * task8b_nnr_model_1024 - for neural regression         
    * task8b_nnc_model_1024 - for neural classification
    * task8b_bertsim_model_32 - for neural classification with BERT 

### topn baseline
```
>>> import task8b
Creating database of vectors word2vec_cnn_200.db
Processing 2020822 words
>>> task8b.bioasq_baseline(test_data='BioASQ-task7bPhaseB-testset1.json')
Processing baseline
Loading BioASQ-task7bPhaseB-testset1.json
Saving results in file bioasq-out-baseline.json
```

### NN regression
```
>>> from classificationneural import bioasq_run
LSTM using sentence length=300
LSTM using embedding dimension=100
Creating database of vectors word2vec_100.db
Processing 2020822 words
Basic summariser
>>> bioasq_run(test_data='BioASQ-task7bPhaseB-testset1.json', model_type='regression', output_filename='bioasq-out-nnr.json')
Running bioASQ
Reading data from BioASQ-training8b.json and rouge_8b.csv
Loading BioASQ-training8b.json
Database word2vec_100.db opened
Loading word embeddings
Word embeddings loaded
Gathering training data
Collecting data
End collecting data
Training BiLSTMSimMul(embed100)+pos-relu(50)
Restoring Similarities model from ./task8b_nnr_model_1024
... output deleted ...
Model restored from file: ./task8b_nnr_model_1024
51/51 [==============================] - 86s 2s/step - loss: 0.0359 - accuracy: 0.0424
Loading BioASQ-task7bPhaseB-testset1.json
LOADED
100% (100 of 100) |####################################| Elapsed Time: 0:05:05 Time:  0:05:05
Saving results in file bioasq-out-nnr.json
>>> 
```

### NN classification

```
>>> from classificationneural import bioasq_run
LSTM using sentence length=300
LSTM using embedding dimension=100
Basic summariser
>>> bioasq_run(test_data='BioASQ-task7bPhaseB-testset1.json', model_type='classification', output_filename='bioasq-out-nnc.json')
Running bioASQ
Reading data from BioASQ-training8b.json and rouge_8b.csv
Loading BioASQ-training8b.json
Setting top 5 classification labels
Database word2vec_100.db opened
Loading word embeddings
Word embeddings loaded
Gathering training data
Collecting data
End collecting data
Training BiLSTMSimMul(embed100)+pos-relu(50)
Restoring Similarities model from ./task8b_nnc_model_1024
... output deleted ...
Model restored from file: ./task8b_nnc_model_1024
51/51 [==============================] - 86s 2s/step - loss: 0.5815 - accuracy: 0.7110
Loading BioASQ-task7bPhaseB-testset1.json
LOADED
100% (100 of 100) |##########################################| Elapsed Time: 0:05:14 Time:  0:05:14
Saving results in file bioasq-out-nnc.json 
>>>
```

###

```
>>> from classificationneural import bioasq_run
LSTM using sentence length=300
LSTM using embedding dimension=100
Basic summariser
>>> bioasq_run(test_data='BioASQ-task7bPhaseB-testset1.json', model_type='bert', output_filename='bioasq-out-bertsim.json')
Running bioASQ
Reading data from BioASQ-training8b.json and rouge_8b.csv
Loading BioASQ-training8b.json
Setting top 5 classification labels
Gathering training data
Collecting data
End collecting data
Training BERTMeanSimMul+pos-relu(50)
Restoring  BERT Similarities model from ./task8b_bertsim_model_32
All model checkpoint weights were used when initializing TFBertModel.

All the weights of TFBertModel were initialized from the model checkpoint at bert-tfmodel-base-uncased.
If your task is similar to the task the model of the ckeckpoint was trained on, you can already use TFBertModel for predictions without further training.




## Examples of cross-validation runs and their results

Below are 10-fold cross-validation results using the BioASQ8b training data.

### Several baseline systems:

```
>>> python task8b.py
```
| Method | Mean SU4 F1 | Std SU4 F1 |
| ------ | --------: | -------: |
| firstn         |  0.26051 | 0.01130 |
| random         |  0.20981 | 0.01409 |
| cos_lsa        |  0.20783 | 0.00852 |
| cos_embeddings |  0.21496 | 0.00488 |

### Neural approaches
#### Classification (NNC)
```
rm diego.out; for F in 1 2 3 4 5 6 7 8 9 10; do python classificationneural.py -m -t LSTMSimilarities --dropout 0.7 --nb_epoch 10 --batch_size 1024 --fold $F >> diego.out; done
```

#### Regression (NNR)
```
rm diego.out; for F in 1 2 3 4 5 6 7 8 9 10; do python classificationneural.py -m -t LSTMSimilarities --dropout 0.7 --nb_epoch 10 --batch_size 1024 --fold $F --regression --rouge_labels >> diego.out; done
```

And similar for the other neural approaches.

For historical reasons, "Dropout" is the ratio of nodes that remain after applying dropout. So, a dropout of 1.0 means that there is no dropout.

| Method | Batch size | Dropout | Epochs | Mean SU4 F1 | Std SU4 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NNR | 1024 | 0.7 | 10 | 0.264 | 0.008 |
| NNC | 1024 | 0.7 | 10 | 0.271 | 0.013 |
| BERT untrained | 32 | 1.0 | 50 | 0.270 | 0.014 |
| BERT trained | 8 | 1.0 | 1 | 0.261 | 0.012 |
| BERT LSTM | 1024 | 0.4 | 10 | **0.274** | 0.010 |
| BioBERT untrained | 1024 | 1.0 | 10 | 0.262 | 0.010 |
| BioBERT LSTM | 1024 | 0.4 | 10 | 0.264 | 0.012 |
| Siamese LSTM | 1024 | 0.8 | 10 | 0.263 | 0.010 |


## Who do I talk to? ###

Diego Molla: [diego.molla-aliod@mq.edu.au](mailto:diego.molla-aliod@mq.edu.au)
