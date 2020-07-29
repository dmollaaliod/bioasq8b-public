"""classification.py -- perform classification-based summarisation using deep learning architectures

Author: Diego Molla <dmollaaliod@gmail.com>
Created: 16/1/2019
"""

import tensorflow as tf # required for running in NCI Raijin
import json
import codecs
import csv
import sys
import os
import shutil
#import re
import random
import glob
from subprocess import Popen, PIPE
from multiprocessing import Pool
from functools import partial
import numpy as np
import progressbar

from sklearn.model_selection import KFold

from nltk import sent_tokenize
from xml_abstract_retriever import getAbstract
from nnmodels import nnc, compare
from summariser.basic import answersummaries
from classification import rouge_to_labels

def bioasq_train(small_data=False, verbose=2, model_type='classification'):
    """Train model for BioASQ"""
    
    assert model_type in ['classification', 'regression', 'bert']
    if model_type == 'classification':
        rouge_labels = False
        classification_type = 'LSTMSimilarities'
        regression = False
        nb_epoch = 10
        dropout = 0.7
        batch_size=1024
        savepath = "./task8b_nnc_model_1024"
    elif model_type == 'regression':
        rouge_labels = True
        classification_type = 'LSTMSimilarities'
        regression = True
        nb_epoch = 10
        dropout = 0.7
        batch_size=1024
        savepath = "./task8b_nnr_model_1024"
    elif model_type == 'bert':
        rouge_labels = False
        classification_type = 'SimilaritiesBERT'
        regression = False
        nb_epoch = 20
        dropout = 1.0
        batch_size=32
        savepath = "./task8b_bertsim_model_32"
        
    if small_data:
        nb_epoch = 3
        
    print("Training for BioASQ", model_type)
    classifier = Classification('BioASQ-training8b.json',
                                'rouge_8b.csv',
                                #'train7b_filtered_v1.json',
                                #'rouge_train7b_dima.csv',
                                rouge_labels=rouge_labels,
                                regression=regression,
                                nb_epoch=nb_epoch,
                                verbose=verbose,
                                classification_type=classification_type,
                                embeddings=True,
                                hidden_layer=50,
                                dropout=dropout,
                                batch_size=batch_size)

    indices = list(range(len(classifier.data)))
    if small_data:
        print("Training bioasq with small data")
        indices = indices[:20]
        nb_epoch = 3
    classifier.train(indices, savepath=savepath)

def bioasq_run(nanswers={"summary": 6,
                         "factoid": 2,
                         "yesno": 2,
                         "list": 3},
#               test_data='BioASQ-trainingDataset6b.json',
#               test_data='BioASQ-training8b.json',
#               test_data='phaseB_5b_01.json',
                test_data='BioASQ-task7bPhaseB-testset1.json',
               model_type='classification',
               output_filename='bioasq-out-nnc.json'):
    """Run model for BioASQ"""
    
    assert model_type in ['classification', 'regression', 'bert']
    if model_type == 'classification':
        classification_type = 'LSTMSimilarities'
        rouge_labels = False
        regression = False
        nb_epoch = 10
        dropout = 0.7
        batch_size=1024
        savepath = "./task8b_nnc_model_1024"
    elif model_type == 'regression':
        classification_type = 'LSTMSimilarities'
        rouge_labels = True
        regression = True
        nb_epoch = 10
        dropout = 0.7
        batch_size=1024
        savepath = "./task8b_nnr_model_1024"
    elif model_type == 'bert':
        classification_type = 'SimilaritiesBERT'
        rouge_labels = False
        regression = False
        nb_epoch = 20
        dropout = 1.0
        batch_size=32
        savepath = "./task8b_bertsim_model_32"

    print("Running bioASQ")
    classifier = Classification('BioASQ-training8b.json',
                                'rouge_8b.csv',
                                nb_epoch=nb_epoch,
                                rouge_labels=rouge_labels,
                                regression=regression,
                                verbose=2,
                                classification_type=classification_type,
                                embeddings=True,
                                hidden_layer=50,
                                dropout=dropout,
                                batch_size=batch_size)
    indices = list(range(len(classifier.data)))
    classifier.train(indices, savepath=savepath, restore_model=True)
    testset = load_test_data(test_data)
    print("LOADED")
    answers = yield_bioasq_answers(classifier,
                                   testset,
                                   nanswers={"summary": 6,
                                             "factoid": 2,
                                             "yesno": 2,
                                             "list": 3})
    result = {"questions": [a for a in answers]}
    print("Saving results in file %s" % output_filename)
    with open(output_filename, 'w') as f:
        f.write(json.dumps(result, indent=2))

def loaddata(filename):
    """Load the JSON data
    >>> data = loaddata('BioASQ-training8b.json')
    Loading BioASQ-training8b.json
    >>> len(data)
    3242
    >>> sorted(data[0].keys())
    ['body', 'concepts', 'documents', 'id', 'ideal_answer', 'snippets', 'type']
    """
    print("Loading", filename)
    data = json.load(open(filename, encoding="utf-8"))
    return [x for x in data['questions'] if 'ideal_answer' in x]

def load_test_data(filename):
    """Load the JSON data
    >>> data = loaddata('BioASQ-training8b.json')
    Loading BioASQ-training8b.json
    >>> len(data)
    3242
    >>> sorted(data[0].keys())
    ['body', 'concepts', 'documents', 'id', 'ideal_answer', 'snippets', 'type']
    """
    print("Loading", filename)
    data = json.load(open(filename, encoding="utf-8"))
    return data['questions']


def yield_candidate_text(questiondata, snippets_only=True):
    """Yield all candidate text for a question
    >>> data = loaddata("BioASQ-training8b.json")
    Loading BioASQ-training8b.json
    >>> y = yield_candidate_text(data[0], snippets_only=True)
    >>> next(y)
    ('55031181e9bde69634000014', 0, 'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes')
    >>> next(y)
    ('55031181e9bde69634000014', 1, "In this study, we review the identification of genes and loci involved in the non-syndromic common form and syndromic Mendelian forms of Hirschsprung's disease.")
    """
    past_pubmed = set()
    sn_i = 0
    for sn in questiondata['snippets']:
        if snippets_only:
            for s in sent_tokenize(sn['text']):
                yield (questiondata['id'], sn_i, s)
                sn_i += 1
            continue

        pubmed_id = os.path.basename(sn['document'])
        if pubmed_id in past_pubmed:
            continue
        past_pubmed.add(pubmed_id)
        file_name = os.path.join("Task6bPubMed", pubmed_id+".xml")
        sent_i = 0
        for s in sent_tokenize(getAbstract(file_name, version="0")[0]):
            yield (pubmed_id, sent_i, s)
            sent_i += 1

def yield_bioasq_answers(classifier, testset, nanswers=3):
    """Yield answer of each record for BioASQ shared task"""
    with progressbar.ProgressBar(max_value=len(testset)) as bar:
        for i, r in enumerate(testset):
            test_question = r['body']
            test_id = r['id']
            test_candidates = [(sent, sentid)
                            for (pubmedid, sentid, sent)
                            in yield_candidate_text(r)]
    #        test_snippet_sentences = [s for snippet in r['snippets']
    #                                  for s in sent_tokenize(snippet['text'])]
            if len(test_candidates) == 0:
                print("Warning: no text to summarise")
                test_summary = ""
            else:
                if isinstance(nanswers,dict):
                    n = nanswers[r['type']]
                else:
                    n = nanswers
                test_summary = " ".join(classifier.answersummaries([(test_question,
                                                                    test_candidates,
                                                                    n)])[0])
                #print("Test summary:", test_summary)

            if r['type'] == "yesno":
                exactanswer = "yes"
            else:
                exactanswer = ""

            yield {"id": test_id,
                "ideal_answer": test_summary,
                "exact_answer": exactanswer}
            bar.update(i)

def collect_one_item(this_index, indices, testindices, data, labels):
    "Collect one item for parallel processing"
    qi, d = this_index
    if qi in indices:
        partition = 'main'
    elif testindices != None and qi in testindices:
        partition = 'test'
    else:
        return None

    this_question = d['body']

    if 'snippets' not in d:
        return None
    data_snippet_sentences = [s for sn in d['snippets']
                              for s in sent_tokenize(sn['text'])]

    if len(data_snippet_sentences) == 0:
        return None

    candidates_questions = []
    candidates_sentences = []
    candidates_sentences_ids = []
    label_data = []
    for pubmed_id, sent_id, sent in yield_candidate_text(d):
        candidates_questions.append(this_question)
        candidates_sentences.append(sent)
        candidates_sentences_ids.append(sent_id)
        label_data.append(labels[(qi, pubmed_id, sent_id)])

    return partition, label_data, candidates_questions, candidates_sentences, candidates_sentences_ids

class BaseClassification:
    """A base classification to be inherited"""
    def __init__(self, corpusFile, rougeFile, metric=['SU4'],
                 rouge_labels=True, labels="topn", labelsthreshold=5):
        """Initialise the classification system."""
        print("Reading data from %s and %s" % (corpusFile, rougeFile))
        self.data = loaddata(corpusFile)
        if not rouge_labels:
            #convert rouge -> label
            self.labels = rouge_to_labels(rougeFile, labels, labelsthreshold, metric=metric)
        else:
            #leave as rouge scores
            self.labels = dict()
            with codecs.open(rougeFile, encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                lineno = 0
                num_errors = 0
                for line in reader:
                    lineno += 1
                    try:
                        key = (int(line['qid']), line['pubmedid'], int(line['sentid']))
                    except:
                        num_errors += 1
                        print("Unexpected error:", sys.exc_info()[0])
                        print("%i %s" % (lineno, str(line).encode('utf-8')))
                    else:
                        self.labels[key] = np.mean([float(line[m]) for m in metric])
                if num_errors > 0:
                    print("%i data items were ignored out of %i because of errors" % (num_errors, lineno))

    def _collect_data_(self, indices, testindices=None):
        """Collect the data given the question indices"""
        print("Collecting data")
        with Pool() as pool:
            collected = pool.map(partial(collect_one_item,
                                         indices=indices,
                                         testindices=testindices,
                                         data=self.data,
                                         labels=self.labels),
                                 enumerate(self.data))
        all_candidates_questions = {'main':[], 'test':[]}
        all_candidates_sentences = {'main':[], 'test':[]}
        all_candidates_sentences_ids = {'main':[], 'test':[]}
        all_labels = {'main':[], 'test':[]}
        for c in collected:
            if c == None:
                continue
            partition, labels_data, candidates_questions, candidates_sentences, candidates_sentences_ids = c
            all_candidates_questions[partition] += candidates_questions
            all_candidates_sentences[partition] += candidates_sentences
            all_candidates_sentences_ids[partition] += candidates_sentences_ids
            all_labels[partition] += labels_data

        print("End collecting data")
        return all_labels, all_candidates_questions, all_candidates_sentences, all_candidates_sentences_ids

class Classification(BaseClassification):
    """A classification system"""
    def __init__(self, corpusFile, rougeFile, metric=['SU4'],
                 rouge_labels=True, labels="topn", labelsthreshold=5,
                 nb_epoch=3, verbose=2,
                 classification_type="Bi-LSTM",
                 embeddings=True,
                 hidden_layer=0,
                 dropout=0.5,
                 regression=False,
                 batch_size=128):
        """Initialise the classification system."""
        BaseClassification.__init__(self, corpusFile, rougeFile, metric=metric,
                                    rouge_labels=rouge_labels, labels=labels, labelsthreshold=labelsthreshold)
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.dropout = dropout

        self.nnc = None
        if classification_type == "BasicNN":
            self.nnc = nnc.BasicNN(embeddings=embeddings,
                                    hidden_layer=hidden_layer,
                                    batch_size=batch_size)
        elif classification_type == "BERT":
            self.nnc = nnc.BasicBERT(hidden_layer=hidden_layer,
                                     batch_size=batch_size)
        elif classification_type == "SimilaritiesBERT":
            self.nnc = nnc.SimilaritiesBERT(hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimMul(),
                                          regression=regression,
                                          positions=True)
        elif classification_type == "LSTMSimilaritiesBERT":
            self.nnc = nnc.LSTMSimilaritiesBERT(hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimMul(),
                                          regression=regression,
                                          positions=True)
        elif classification_type == "SimilaritiesBioBERT":
            self.nnc = nnc.SimilaritiesBioBERT(hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimMul(),
                                          regression=regression,
                                          positions=True)
        elif classification_type == "LSTMSimilaritiesBioBERT":
            self.nnc = nnc.LSTMSimilaritiesBioBERT(hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimMul(),
                                          regression=regression,
                                          positions=True)
        elif classification_type == "SimilaritiesBERTTrainable":
            self.nnc = nnc.SimilaritiesBERT(hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimMul(),
                                          regression=regression,
                                          positions=True,
                                          trainable=True)
        elif classification_type == "Similarities":
            self.nnc = nnc.Similarities(embeddings=embeddings,
                                          hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimMul(),
                                          regression=regression,
                                          positions=True)
        elif classification_type == "SimilaritiesYu":
            self.nnc = nnc.Similarities(embeddings=embeddings,
                                          hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimYu(),
                                          regression=regression,
                                          positions=True)
        elif classification_type == "SimilaritiesEuc":
            self.nnc = nnc.Similarities(embeddings=embeddings,
                                          hidden_layer=hidden_layer,
                                          batch_size=batch_size,
                                          comparison=compare.SimEuc(),
                                          regression=regression,
                                          positions=True)
        elif classification_type == "CNNSimilarities":
            self.nnc = nnc.CNNSimilarities(embeddings=embeddings,
                                             hidden_layer=hidden_layer,
                                             batch_size=batch_size,
                                             comparison = compare.SimMul())
        elif classification_type == "CNNSimilaritiesYu":
            self.nnc = nnc.CNNSimilarities(embeddings=embeddings,
                                             hidden_layer=hidden_layer,
                                             batch_size=batch_size,
                                             comparison = compare.SimYu())
        elif classification_type == "CNNSimilaritiesEuc":
            self.nnc = nnc.CNNSimilarities(embeddings=embeddings,
                                             hidden_layer=hidden_layer,
                                             batch_size=batch_size,
                                             comparison = compare.SimEuc())
        elif classification_type == "LSTMSimilarities":
            self.nnc = nnc.LSTMSimilarities(embeddings=embeddings,
                                              hidden_layer=hidden_layer,
                                              batch_size=batch_size,
                                              comparison = compare.SimMul(),
                                              regression=regression,
                                              positions=True)
        elif classification_type == "LSTMSiameseSimilarities":
            self.nnc = nnc.LSTMSimilarities(embeddings=embeddings,
                                              hidden_layer=hidden_layer,
                                              batch_size=batch_size,
                                              comparison = compare.SimMul(),
                                              regression=regression,
                                              positions=True,
                                              siamese=True)
        elif classification_type == "LSTMSimilaritiesYu":
            self.nnc = nnc.LSTMSimilarities(embeddings=embeddings,
                                              hidden_layer=hidden_layer,
                                              batch_size=batch_size,
                                              comparison = compare.SimMul(),
                                              regression=regression,
                                              positions=True)
        elif classification_type == "LSTMSimilaritiesEuc":
            self.nnc = nnc.LSTMSimilarities(embeddings=embeddings,
                                              hidden_layer=hidden_layer,
                                              batch_size=batch_size,
                                              comparison = compare.SimEuc(),
                                              regression=regression,
                                              positions=True)

    def extractfeatures(self, questions, candidates_sentences):
        """ Return the features"""
        assert len(questions) == len(candidates_sentences)

        return ([self.nnc.cleantext(sentence) for sentence in candidates_sentences],
                [self.nnc.cleantext(question) for question in questions])


    def train(self, indices, testindices=None, foldnumber=0, restore_model=False,
              savepath=None,
              save_test_predictions=None):
        """Train the classifier given the question indices"""
        print("Gathering training data")
        if savepath is None:
            savepath="savedmodels/%s_%i" % (self.nnc.name(), foldnumber)
        all_labels, candidates_questions, candidates_sentences, candidates_sentences_ids = \
        self._collect_data_(indices, testindices)

        features = self.extractfeatures(candidates_questions['main'],
                                        candidates_sentences['main'])

        print("Training %s" % self.nnc.name())
        if testindices == None:
            validation_data = None
        else:
            features_test = self.extractfeatures(candidates_questions['test'],
                                                 candidates_sentences['test'])
            validation_data = (features_test[0], features_test[1],
                               [[r] for r in all_labels['test']],
                               [[cid] for cid in candidates_sentences_ids['test']])
        loss_history = self.nnc.fit(features[0], features[1],
                                          np.array([[r] for r in all_labels['main']]),
                                          X_positions=np.array([[cid] for cid in candidates_sentences_ids['main']]),
                                          validation_data = validation_data,
                                          nb_epoch=self.nb_epoch,
                                          verbose=self.verbose,
                                          dropoutrate=self.dropout,
                                          savepath=savepath,
                                          restore_model=restore_model)        
        if save_test_predictions:
            predictions_test = self.nnc.predict(features_test[0],
                                                features_test[1],
                                                X_positions=np.array([[cid] for cid in candidates_sentences_ids['test']]))
            predictions_test = [p[0] for p in predictions_test]
            print("Saving predictions in %s" % save_test_predictions)
            with open(save_test_predictions, "w") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "target", "prediction"])
                writer.writeheader()
                for i, p in enumerate(predictions_test):
                    writer.writerow({"id": i,
                                     "target": all_labels['test'][i],
                                     "prediction": p})
                print("Predictions saved")

        if restore_model:
            return

        if testindices:
            return loss_history['loss'][-1], loss_history['val_loss'][-1]
        else:
            return loss_history['loss'][-1]

    def test(self, indices):
        """Test the classifier given the question indices"""
        print("Gathering test data")
        all_labels, candidates_questions, candidates_sentences, candidates_sentences_ids = \
        self._collect_data_(indices)

        features = self.extractfeatures(candidates_questions['main'],
                                        candidates_sentences['main'])

        print("Testing NNC")
        loss = self.nnc.test(features[0], 
                              features[1],
                              [[r] for r in all_labels['main']],
                              X_positions=[[cid] for cid in candidates_sentences_ids['main']])
        print("Loss = %f" % loss)
        return loss

    def answersummaries(self, questions_and_candidates, beamwidth=0):
        if beamwidth > 0:
            print("Beam width is", beamwidth)
            return answersummaries(questions_and_candidates, self.extractfeatures, self.nnc.predict, beamwidth)
        else:
            return answersummaries(questions_and_candidates, self.extractfeatures, self.nnc.predict)

    def answersummary(self, question, candidates_sentences,
                      n=3, qindex=None):
        """Return a summary that answers the question

        qindex is not used but needed for compatibility with oracle"""
        return self.answersummaries((question, candidates_sentences, n))

def evaluate_one(di, dataset, testindices, nanswers, rougepath):
    """Evaluate one question"""
    if di not in testindices:
        return None
    question = dataset[di]['body']
    if 'snippets' not in dataset[di].keys():
        return None
    candidates = [(sent, sentid) for (pubmedid, sentid, sent) in yield_candidate_text(dataset[di])]
    if len(candidates) == 0:
        # print("Warning: No text to summarise; ignoring this text")
        return None

    if type(nanswers) == dict:
        n = nanswers[dataset[di]['type']]
    else:
        n = nanswers
    rouge_text = """<EVAL ID="%i">
 <MODEL-ROOT>
 %s/models
 </MODEL-ROOT>
 <PEER-ROOT>
 %s/summaries
 </PEER-ROOT>
 <INPUT-FORMAT TYPE="SEE">
 </INPUT-FORMAT>
""" % (di, rougepath, rougepath)
    rouge_text += """ <PEERS>
  <P ID="A">summary%i.txt</P>
 </PEERS>
 <MODELS>
""" % (di)

    if type(dataset[di]['ideal_answer']) == list:
        ideal_answers = dataset[di]['ideal_answer']
    else:
        ideal_answers = [dataset[di]['ideal_answer']]

    for j in range(len(ideal_answers)):
        rouge_text += '  <M ID="%i">ideal_answer%i_%i.txt</M>\n' % (j,di,j)
        with codecs.open(rougepath + '/models/ideal_answer%i_%i.txt' % (di,j),
                         'w', 'utf-8') as fout:
            a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</body>\n</html>'.format(ideal_answers[j])
            fout.write(a+'\n')
    rouge_text += """ </MODELS>
</EVAL>
"""
    target = {'id': dataset[di]['id'],
              'ideal_answer': ideal_answers,
              'exact_answer': ""}
    return rouge_text, di, (question, candidates, n), target

def evaluate(classificationClassInstance, rougeFilename="rouge.xml", nanswers=3,
             tmppath='', load_models=False, small_data=False, fold=0):
    """Evaluate a classification-based summariser

    nanswers is the number of answers. If it is a dictionary, then the keys indicate the question type, e.g.
    nanswers = {"summary": 6,
                "factoid": 2,
                "yesno": 2,
                "list": 3}
"""
    if tmppath == '':
        modelspath = 'saved_models_Similarities'
        rougepath = '../rouge'
        crossvalidationpath = 'crossvalidation'
    else:
        modelspath = tmppath + '/saved_models'
        rougepath = tmppath + '/rouge'
        crossvalidationpath = tmppath + '/crossvalidation'
        rougeFilename = rougepath + "/" + rougeFilename
        if not os.path.exists(rougepath):
            os.mkdir(rougepath)
        for f in glob.glob(rougepath + '/*'):
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
            else:
                print("Warning: %f is neither a file nor a directory" % (f))
        os.mkdir(rougepath + '/models')
        os.mkdir(rougepath + '/summaries')
        if not os.path.exists(crossvalidationpath):
            os.mkdir(crossvalidationpath)

    dataset = classificationClassInstance.data
    indices = [i for i in range(len(dataset))
               #if dataset[i]['type'] == 'summary'
               #if dataset[i]['type'] == 'factoid'
               #if dataset[i]['type'] == 'yesno'
               #if dataset[i]['type'] == 'list'
               ]
    if small_data:
        indices = indices[:100]

    random.seed(1234)
    random.shuffle(indices)

    rouge_results = []
    rouge_results_P = []
    rouge_results_R = []
    the_fold = 0
    kf = KFold(n_splits=10)
    for (traini, testi) in kf.split(indices):
        the_fold += 1

        if fold > 0 and the_fold != fold:
            continue

        if small_data and the_fold > 2:
           break

        print("Cross-validation Fold %i" % the_fold)
        trainindices = [indices[i] for i in traini]
        testindices = [indices[i] for i in testi]

        save_test_predictions = crossvalidationpath + "/test_results_%i.csv" % the_fold
        (trainloss,testloss) = classificationClassInstance.train(trainindices,
                                                             testindices,
                                                             foldnumber=the_fold,
                                                             restore_model=load_models,
                                                             savepath="%s/saved_model_%i" % (modelspath, the_fold),
                                                             save_test_predictions=save_test_predictions)

        for f in glob.glob(rougepath+'/models/*')+glob.glob(rougepath+'/summaries/*'):
            os.remove(f)

        with open(rougeFilename,'w') as frouge:
           print("Collecting evaluation results")
           frouge.write('<ROUGE-EVAL version="1.0">\n')
           #with Pool() as pool:
           #    evaluation_data = \
           #       pool.map(partial(evaluate_one,
           #                        dataset=dataset,
           #                        testindices=testindices,
           #                        nanswers=nanswers,
           #                        rougepath=rougepath),
           #                range(len(dataset)))

           evaluation_data = [evaluate_one(i,
                                           dataset=dataset,
                                           testindices=testindices,
                                           nanswers=nanswers,
                                           rougepath=rougepath)
                              for i in range(len(dataset))]


           summaries = classificationClassInstance.answersummaries([e[2] for e in evaluation_data if e != None])

           eval_test_system = []
           eval_test_target = []
           for data_item in evaluation_data:
               if data_item == None:
                   continue
               rouge_item, di, system_item, target_item = data_item
               summary = summaries.pop(0)
               #print(di)
               with codecs.open(rougepath+'/summaries/summary%i.txt' % (di),
                               'w', 'utf-8') as fout:
                   a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</body>\n</html>'.format(" ".join(summary))
                   fout.write(a+'\n')
                   # fout.write('\n'.join([s for s in summary])+'\n')

               frouge.write(rouge_item)
               system_item = {'id': dataset[di]['id'],
                              'ideal_answer': " ".join(summary),
                              'exact_answer': ""}

               eval_test_system.append(system_item)
               eval_test_target.append(target_item)

           assert len(summaries) == 0

           frouge.write('</ROUGE-EVAL>\n')

        json_summaries_file = crossvalidationpath + "/crossvalidation_%i_summaries.json" % the_fold
        print("Saving summaries in file %s" % json_summaries_file)
        with open(json_summaries_file,'w') as fcv:
            fcv.write(json.dumps({'questions': eval_test_system}, indent=2))
        json_gold_file = crossvalidationpath + "/crossvalidation_%i_gold.json" % the_fold
        print("Saving gold data in file %s" % json_gold_file)
        with open(json_gold_file,'w') as fcv:
            fcv.write(json.dumps({'questions': eval_test_target}, indent=2))

        print("Calling ROUGE", rougeFilename)
        ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ' \
            + '../rouge/RELEASE-1.5.5/data -c 95 -2 4 -u -x -n 4 -a ' \
            + rougeFilename
        stream = Popen(ROUGE_CMD, shell=True, stdout=PIPE).stdout
        lines = stream.readlines()
        stream.close()
        for l in lines:
            print(l.decode('ascii').strip())
        print()

        F = {'N-1':float(lines[3].split()[3]),
             'N-2':float(lines[7].split()[3]),
             'L':float(lines[11].split()[3]),
             'S4':float(lines[15].split()[3]),
             'SU4':float(lines[19].split()[3]),
             'trainloss':trainloss,
             'testloss':testloss}
        P = {'N-1':float(lines[2].split()[3]),
             'N-2':float(lines[6].split()[3]),
             'L':float(lines[10].split()[3]),
             'S4':float(lines[14].split()[3]),
             'SU4':float(lines[18].split()[3]),
             'trainloss':trainloss,
             'testloss':testloss}
        R = {'N-1':float(lines[1].split()[3]),
             'N-2':float(lines[5].split()[3]),
             'L':float(lines[9].split()[3]),
             'S4':float(lines[15].split()[3]),
             'SU4':float(lines[17].split()[3]),
             'trainloss':trainloss,
             'testloss':testloss}
        rouge_results.append(F)
        rouge_results_P.append(P)
        rouge_results_R.append(R)

        print("F N-2: %1.5f SU4: %1.5f TrainLoss: %1.5f TestLoss: %1.5f" % (
               F['N-2'], F['SU4'], F['trainloss'], F['testloss']
        ))
        print("P N-2: %1.5f SU4: %1.5f TrainLoss: %1.5f TestLoss: %1.5f" % (
               P['N-2'], P['SU4'], P['trainloss'], P['testloss']
        ))
        print("R N-2: %1.5f SU4: %1.5f TrainLoss: %1.5f TestLoss: %1.5f" % (
               R['N-2'], R['SU4'], R['trainloss'], R['testloss']
        ))


    print("%5s %7s %7s %7s %7s" % ('', 'N-2', 'SU4', 'TrainLoss', 'TestLoss'))
    for i in range(len(rouge_results)):
        print("%5i %1.5f %1.5f %1.5f %1.5f" % (i+1,rouge_results[i]['N-2'],rouge_results[i]['SU4'],
                                       rouge_results[i]['trainloss'],rouge_results[i]['testloss']))
    mean_N2 = np.average([rouge_results[i]['N-2']
                          for i in range(len(rouge_results))])
    mean_SU4 = np.average([rouge_results[i]['SU4']
                           for i in range(len(rouge_results))])
    mean_N2_P = np.average([rouge_results_P[i]['N-2']
                          for i in range(len(rouge_results_P))])
    mean_SU4_P = np.average([rouge_results_P[i]['SU4']
                           for i in range(len(rouge_results_P))])
    mean_N2_R = np.average([rouge_results_R[i]['N-2']
                          for i in range(len(rouge_results_R))])
    mean_SU4_R = np.average([rouge_results_R[i]['SU4']
                           for i in range(len(rouge_results_R))])
    mean_Trainloss = np.average([rouge_results[i]['trainloss']
                                 for i in range(len(rouge_results))])
    mean_Testloss = np.average([rouge_results[i]['testloss']
                                for i in range(len(rouge_results))])
    print("%5s %1.5f %1.5f %1.5f %1.5f" % ("mean",mean_N2,mean_SU4,mean_Trainloss,mean_Testloss))
    stdev_N2 = np.std([rouge_results[i]['N-2']
                       for i in range(len(rouge_results))])
    stdev_SU4 = np.std([rouge_results[i]['SU4']
                        for i in range(len(rouge_results))])
    stdev_N2_P = np.std([rouge_results_P[i]['N-2']
                       for i in range(len(rouge_results_P))])
    stdev_SU4_P = np.std([rouge_results_P[i]['SU4']
                        for i in range(len(rouge_results_P))])
    stdev_N2_R = np.std([rouge_results_R[i]['N-2']
                       for i in range(len(rouge_results_R))])
    stdev_SU4_R = np.std([rouge_results_R[i]['SU4']
                        for i in range(len(rouge_results_R))])
    stdev_Trainloss = np.std([rouge_results[i]['trainloss']
                              for i in range(len(rouge_results))])
    stdev_Testloss = np.std([rouge_results[i]['testloss']
                             for i in range(len(rouge_results))])
    print("%5s %1.5f %1.5f %1.5f %1.5f" % ("stdev",stdev_N2,stdev_SU4,stdev_Trainloss,stdev_Testloss))
    print()
    return mean_SU4, stdev_SU4, mean_SU4_P, stdev_SU4_P, mean_SU4_R, stdev_SU4_R, mean_Testloss, stdev_Testloss

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    #import sys
    #bioasq_train(small_data=True, model_type='classification')
    #bioasq_train(small_data=True, model_type='regression')
    #bioasq_train(small_data=True, model_type='bert')
    #bioasq_train(model_type='classification')
    #bioasq_train(model_type='regression')
    #bioasq_train(model_type='bert')
    #print ("SAVED MODEL - NOW TRY RUNNING IT")
    #bioasq_run(model_type='classification', output_filename='bioasq-out-nnc.json')
    #bioasq_run(model_type='regression', output_filename='bioasq-out-nnr.json')
    #bioasq_run(model_type='bert', output_filename='bioasq-out-bert.json')
    #sys.exit()

    import argparse
    import time
    import socket
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--nb_epoch', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help="Verbosity level")
    parser.add_argument('-t', '--classification_type',
                        choices=("BasicNN", # "CNN", "LSTM", "Bi-LSTM",
                                 "BERT", "SimilaritiesBERT", "SimilaritiesBERTTrainable",
                                 "LSTMSimilaritiesBERT", "LSTMSimilaritiesBioBERT",
                                 "SimilaritiesBioBERT",
                                 "Similarities", "CNNSimilarities", "LSTMSimilarities",
                                 "LSTMSiameseSimilarities",
                                 "SimilaritiesYu", "CNNSimilaritiesYu", "LSTMSimilaritiesYu",
                                 "SimilaritiesEuc", "CNNSimilaritiesEuc", "LSTMSimilaritiesEuc"),
                        default="Similarities",
                        help="Type of classification")
    parser.add_argument('-S', '--small', action="store_true",
                        help='Run on a small subset of the data')
    parser.add_argument('-l', '--load', action="store_true",
                        help='load pre-trained model')
    parser.add_argument('-m', '--embeddings', action="store_true",
                        help="Use pre-trained embeddings")
    parser.add_argument('-d', '--hidden_layer', type=int, default=50,
                        help="Size of the hidden layer (0 if there is no hidden layer)")
    parser.add_argument('-r', '--dropout', type=float, default=0.9,
                        help="Keep probability for the dropout layers")
    parser.add_argument('-a', '--tmppath', default='',
                        help="Path for temporary data and files")
    parser.add_argument('-n', '--rouge_labels', default=False,action='store_true',
                        help="Use raw rouge scores as labels. If not specified labels are converted to True/False categories")
    parser.add_argument('-g', '--regression', default=False,action='store_true',
                        help="Use regression model (linear activation) instead of classification model (sigmoid activation)")
    parser.add_argument('-s', '--batch_size', type=int, default=4096,
                        help="Batch size for gradient descent")
    parser.add_argument('-c', '--svd_components', type=int, default=100,
                        help="Depth of the LSTM stack")
    parser.add_argument("-f", "--fold", type=int, default=0,
                        help="Use only the specified fold (0 for all folds)")

    args = parser.parse_args()

    if args.tmppath != '':
        nnc.DB = "%s/%s" % (args.tmppath, nnc.DB)

    print("rouge_labels: %s" % args.rouge_labels)
    classifier = Classification('BioASQ-training8b.json',
                           'rouge_8b.csv',
                           #'train7b_filtered_v3.json',
                           #'rouge_train7b_dima_v3.csv',
                           rouge_labels=args.rouge_labels,
                           regression=args.regression,
                           nb_epoch=args.nb_epoch,
                           verbose=args.verbose,
                           classification_type=args.classification_type,
                           embeddings=args.embeddings,
                           hidden_layer=args.hidden_layer,
                           dropout=args.dropout,
                           batch_size=args.batch_size)

    print("%s with epochs=%i and batch size=%i" % (classifier.nnc.name(), 
                                                   args.nb_epoch,
                                                   args.batch_size))


    mean_SU4, stdev_SU4, mean_SU4_P, stdev_SU4_P, mean_SU4_R, stdev_SU4_R, mean_Testloss, stdev_Testloss = \
                              evaluate(classifier,
                                       nanswers={"summary": 6,
                                                 "factoid": 2,
                                                 "yesno": 2,
                                                 "list": 3},
                                       tmppath=args.tmppath,
                                       load_models=args.load,
                                       small_data=args.small,
                                       fold = args.fold)
    end_time = time.time()
    elapsed = time.strftime("%X", time.gmtime(end_time - start_time))
    print("Time elapsed: %s" % (elapsed))
    print("| Type | Fold | Epochs | Dropout | meanSU4 | stdevSU4 | meanSU4_P | stdevSU4_P | meanSU4_R | stdevSU4_R | meanTestLoss | stdevTestLoss | Time | Hostname |")
    print("| %s | %i | %i | %f | %f | %f | %f | %f | %f | %f | %f | %f | %s | %s |" % \
               (classifier.nnc.name(),
                args.fold,
                args.nb_epoch,
                args.dropout,
                mean_SU4,
                stdev_SU4,
                mean_SU4_P,
                stdev_SU4_P,
                mean_SU4_R,
                stdev_SU4_R,
                mean_Testloss,
                stdev_Testloss,
                elapsed,
                socket.gethostname()))

