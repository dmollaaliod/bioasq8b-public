"""regression.py - Regression-based summariser"""

import tensorflow as tf # Import tensorflow first to avoid problems
                        # with dependencies in the NCI cluster Raijin
import json
import os
import csv
import codecs
import random
from subprocess import Popen, PIPE
#import glob
import pickle
import sys

SENTENCE_LENGTH = 60
PUBMED_FILES_PATH = "Task6bPubMed"

import numpy as np
import scipy.sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_distances

from nltk import sent_tokenize

from my_tokenizer import my_tokenize
import word2vec

from xml_abstract_retriever import getAbstract

from itertools import product

def gridregression(idx=None,
                   C=(0.1, 1, 10, 100, 1000),
                   gamma=(0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001,
                          0.0000001, 0.00000001),
                   kernel=('poly5','poly7')):
    """Grid search on parameters for SVR

    if idx != None, do only the iteration set by the index"""
    if idx == None:
        with open('grid_regression.csv','w') as f:
            f.write('Kernel,C,Gamma,MeanSU4,StdevSU4\n')
        for current_kernel, current_C, current_gamma in product(kernel, C, gamma):
            print("Grid search with kernel=%s, C=%f, gamma=%f" % (current_kernel,
                                                                  current_C,
                                                                  current_gamma))
            regressor = Regression('golden_13.json',
                                   'golden_13PubMed.csv',
                                   kernel=current_kernel,
                                   C=current_C,
                                   gamma=current_gamma)
            mean_SU4, stdev_SU4 = evaluate(regressor,
                                           nanswers = {"summary": 6,
                                               "factoid": 2,
                                               "yesno": 2,
                                               "list": 3})
            with open('grid_regression.csv','a') as f:
                f.write('%s,%f,%f,%f,%f\n' % (current_kernel,
                                              current_C,
                                              current_gamma,
                                              mean_SU4,
                                              stdev_SU4))
    else:
        all_items = [tuples for tuples in product(kernel, C, gamma)]
        current_kernel, current_C, current_gamma = all_items[idx]
        print("Grid search %i with kernel=%s, C=%f and gamma=%f" % (idx,
                                                                    current_kernel,
                                                                    current_C,
                                                                    current_gamma))
        regressor = Regression('golden_13.json',
                               'golden_13PubMed.csv',
                               kernel=current_kernel,
                               C=current_C,
                               gamma=current_gamma)
        mean_SU4, stdev_SU4 = evaluate(regressor,
                                       nanswers = {"summary": 6,
                                           "factoid": 2,
                                           "yesno": 2,
                                           "list": 3})
        print('Index,Kernel,C,Gamma,MeanSU4,StdevSU4')
        print('%i,%s,%f,%f,%f,%f' % (idx,
                                     current_kernel,
                                     current_C,
                                     current_gamma,
                                     mean_SU4,
                                     stdev_SU4))

def bioasq(fitted_regression_class_instance,
           nanswers={"summary": 6,
                     "factoid": 2,
                     "yesno": 2,
                     "list": 3},
           test_data='phaseB_3b_01.json',
           output_filename='bioasq-out-regression.json'):
    """Produce results ready for submission to BioASQ"""
    print("Processing test data")
    testset = loaddata(test_data)
    answers = yield_bioasq_answers(fitted_regression_class_instance,
                                   testset,
                                   nanswers=nanswers)
    result = {"questions":[a for a in answers]}
    print("Saving results in file %s" % output_filename)
    with open(output_filename, 'w') as f:
        f.write(json.dumps(result, indent=2))


def yield_bioasq_answers(regressor, testset,
                         nanswers = {"summary": 6,
                                     "factoid": 2,
                                     "yesno": 2,
                                     "list": 3}):
    """Yield answer of each record for BioASQ shared task"""
    for r in testset:
        test_question = r['body']
        test_id = r['id']
        test_candidates = [(sent, sentid)
                           for (pubmedid, sentid, sent)
                           in yield_candidate_text(r)]
        test_snippet_sentences = [s for snippet in r['snippets']
                                  for s in sent_tokenize(snippet['text'])]
        if len(test_candidates) == 0:
            print("Warning: no text to summarise")
            test_summary = ""
        else:
            if isinstance(nanswers,dict):
                n = nanswers[r['type']]
            else:
                n = nanswers
            test_summary = " ".join(
                regressor.answersummary(test_question,
                                        test_candidates,
                                        test_snippet_sentences,
                                        n=n)
            )
        if r['type'] == "yesno":
            exactanswer = "yes"
        else:
            exactanswer = ""

        yield {"id": test_id,
               "ideal_answer": test_summary,
               "exact_answer": exactanswer}

def loaddata(filename):
    """Load the JSON data
    >>> data = loaddata('BioASQ-trainingDataset6b.json')
    >>> len(data)
    2251
    >>> sorted(data[0].keys())
    ['body', 'concepts', 'documents', 'id', 'ideal_answer', 'snippets', 'type']
    """
    data = json.load(open(filename, encoding='utf-8'))
    return [x for x in data['questions'] if 'ideal_answer' in x]

def yield_candidate_text(questiondata, snippets_only=True):
    """Yield all candidate text for a question
    >>> data = loaddata("BioASQ-trainingDataset6b.json")
    >>> y = yield_candidate_text(data[0], snippets_only=True)
    >>> next(y)
    ('55031181e9bde69634000014', 0, 'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes')
    >>> next(y)
    ('55031181e9bde69634000014', 1, "In this study, we review the identification of genes and loci involved in the non-syndromic common form and syndromic Mendelian forms of Hirschsprung's disease.")
    """
    pastpubmed = set()
    sn_i = 0
    for sn in questiondata['snippets']:
        if snippets_only:
            for s in sent_tokenize(sn['text']):
                yield (questiondata['id'], sn_i, s)
                sn_i += 1
            continue

        pubmedid = os.path.basename(sn['document'])
        if pubmedid in pastpubmed:
            continue
        pastpubmed.add(pubmedid)
        filename = os.path.join(PUBMED_FILES_PATH, pubmedid+".xml")
        senti = 0
        for s in sent_tokenize(getAbstract(filename, version="0")[0]):
            yield (pubmedid, senti, s)
            senti += 1


def yieldRouge(CorpusFile, xml_rouge_filename="rouge.xml",
               snippets_only=True):
    """yield ROUGE scores of all sentences in corpus
    >>> rouge = yieldRouge('BioASQ-trainingDataset6b.json')
    >>> target = (0, '55031181e9bde69634000014', 0, {'SU4': 0.09399, 'L': 0.04445, 'N-1': 0.31915, 'S4': 0.02273, 'N-2': 0.13043}, 'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes')
    >>> next(rouge) == target
    True
    >>> target2 = (0, '55031181e9bde69634000014', 1, {'SU4': 0.2, 'L': 0.09639, 'N-1': 0.41379, 'S4': 0.04938, 'N-2': 0.18823}, "In this study, we review the identification of genes and loci involved in the non-syndromic common form and syndromic Mendelian forms of Hirschsprung's disease.")
    >>> next(rouge) == target2
    True
    """
    data = loaddata(CorpusFile)
    for qi in range(len(data)):
        ai = 0
        if type(data[qi]['ideal_answer']) == list:
            ideal_answers = data[qi]['ideal_answer']
        else:
            ideal_answers = [data[qi]['ideal_answer']]
        for answer in ideal_answers:
            modelfilename = os.path.join('..', 'rouge', 'models', 'model_'+str(ai))
            with codecs.open(modelfilename,'w','utf-8') as fout:
                a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</a></body>\n</html>'.format(answer)
                fout.write(a + "\n")
            ai += 1
        if 'snippets' not in data[qi].keys():
            print("Warning: No snippets in question: %s" % data[qi]['body'])
            continue
#        for qsnipi in range(len(data[qi]['snippets'])):
#            text = data[qi]['snippets'][qsnipi]['text']
        for (pubmedid,senti,sent) in yield_candidate_text(data[qi],
                                                          snippets_only):
            #text = data[qi]['snippets'][qsnipi]['text']
            #senti = -1
            #for sent in sent_tokenize(text):
            #    senti += 1
                with open(xml_rouge_filename,'w') as rougef:
                    rougef.write("""<ROUGE-EVAL version="1.0">
 <EVAL ID="1">
 <MODEL-ROOT>
 ../rouge/models
 </MODEL-ROOT>
 <PEER-ROOT>
 ../rouge/summaries
 </PEER-ROOT>
 <INPUT-FORMAT TYPE="SEE" />
 <PEERS>
   <P ID="A">summary_1</P>
 </PEERS>
 <MODELS>
""")
                    for ai in range(len(ideal_answers)):
                        rougef.write(  '<M ID="%i">model_%i</M>\n' % (ai,ai))
                    rougef.write("""</MODELS>
</EVAL>
</ROUGE-EVAL>""")

                peerfilename = os.path.join('..', 'rouge', 'summaries', 'summary_1')
                with codecs.open(peerfilename,'w', 'utf-8') as fout:
                    a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</a></body>\n</html>'.format(sent)
                    fout.write(a + '\n')
                ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ../rouge/RELEASE-1.5.5/data -c 95 -2 4 -u -x -n 4 -a ' + xml_rouge_filename
                # ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ../rouge/RELEASE-1.5.5/data -a -n 2 -2 4 -U %s' % (xml_rouge_filename)
                stream = Popen(ROUGE_CMD, shell=True, stdout=PIPE).stdout
                lines = stream.readlines()
                stream.close()
                F = {'N-1':float(lines[3].split()[3]),
                     'N-2':float(lines[7].split()[3]),
                     'L':float(lines[11].split()[3]),
                     'S4':float(lines[15].split()[3]),
                     'SU4':float(lines[19].split()[3])}
#                yield (qi,qsnipi,senti,F,sent)
                yield (qi,pubmedid,senti,F,sent)

def saveRouge(corpusfile, outfile, snippets_only=True):
    "Compute and save the ROUGE scores of the individual snippet sentences"
    with open(outfile,'w') as f:
        writer = csv.writer(f)
#        writer.writerow(('qid','snipid','sentid','N1','N2','L','S4','SU4','sentence text'))
        writer.writerow(('qid', 'pubmedid', 'sentid', 'N1', 'N2', 'L', 'S4', 'SU4', 'sentence text'))
        for (qi, qsnipi, senti, F, sent) in yieldRouge(corpusfile, snippets_only=snippets_only):
            writer.writerow((qi, qsnipi, senti, F['N-1'], F['N-2'], F['L'], F['S4'], F['SU4'], sent))

class BaseRegression:
    "A base class for regression systems"
    def __init__(self, corpusFile, rougeFile, add_snippets=False,
                 embedding_distances=True,
                 metric=["SU4"]):
        """Initialise the regression system.

        metric is a list of possible ROUGE metrics. If there are more
        than one, the target metric is the average of the metrics
        listed.
        """
        print("Reading data from %s and %s" % (corpusFile, rougeFile))
        self.add_snippets = add_snippets
        self.embedding_distances = embedding_distances
        self.data = loaddata(corpusFile)
        self.rouge = dict()
        with codecs.open(rougeFile, 'r', 'utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            index = [header.index(m) for m in metric]
            for line in reader:
                try:
                    key = (int(line[0]),line[1],int(line[2]))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    print([l.encode('utf-8') for l in line])
                else:
                    self.rouge[key] = np.mean([float(line[i]) for i in index])

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            cls = pickle.load(f)
        return cls

    # def extractfeatures(self,questions,candidatesentences,snippets=[]):
    #     """Return the features - distance, min(distance to a snippet), and tfidf
    #       questions - the questions
    #       candidate sentences - a list of sentences
    #       snippets - a list of retrieved snippets if available, each element is a tuple
    #           (numberofcandidates,snippetsofthesecandidates)
    #     The length of the two lists questions and candidatesentences must be equal"""

    #     assert len(questions) == len(candidatesentences)
    #     featurescandidates = self.tfidf.transform(candidatesentences)
    #     featuresquestions = self.tfidf.transform(questions)
    #     similarities = [pairwise_distances(featuresquestions.getrow(i),
    #                                        featurescandidates.getrow(i),
    #                                        'cosine')[0]
    #                     for i in range(len(candidatesentences))]
    #     if len(snippets) == 0:
    #         minsnippetdistances = [[0.5]]*len(candidatesentences)
    #     else:
    #         assert len(candidatesentences) == sum([n for (n,sn) in snippets])
    #         featuressnippets = []
    #         for (n,sn) in snippets:
    #             auxtfidf = self.tfidf.transform(sn)
    #             for i in range(n):
    #                 featuressnippets.append(auxtfidf)
    #         minsnippetdistances = [[np.amin(pairwise_distances(featurescandidates.getrow(i),
    #                                                            featuressnippets[i],
    #                                                            'cosine'))]
    #                                for i in range(len(candidatesentences))]
    #     return scipy.sparse.hstack((similarities,minsnippetdistances,featurescandidates))

    def extractfeatures(self, questions, candidatesentences, candidatessentences_ids=[],
                        snippets=[]):
        """Return the features - based on distance metrics between question and text
          questions - the questions
          candidatesentences - a list of sentences
          snippets - a list of retrieved snippets if available; there is a list of snippets per candidate sentence
        The length of the two lists must be equal"""

        assert len(questions) == len(candidatesentences)

        # cosine distance between question and sentences
        print("extractfeatures [1    ]\r", end='')
        sys.stdout.flush()
        tfidfsnippets = self.tfidf.transform(candidatesentences)
        print("extractfeatures [12   ]\r", end='')
        sys.stdout.flush()
        tfidfquestions = self.tfidf.transform(questions)
        print("extractfeatures [123  ]\r", end='')
        sys.stdout.flush()
        tfidfdistances = [pairwise_distances(tfidfquestions.getrow(i),
                                             tfidfsnippets.getrow(i),
                                             'cosine')[0]
                          for i in range(len(candidatesentences))]

        print("extractfeatures [1234 ]\r", end='')
        sys.stdout.flush()
        tfidfvocabulary = self.tfidf.get_feature_names()

        # minimum distance to a snippet
        if len(snippets) == 0 or not self.add_snippets:
            minsnippetdistances = [[0.5]]*len(candidatesentences)
            bestsnippets = [[0]]*len(candidatesentences)
        else:
            assert len(candidatesentences) == len(snippets)
            featuressnippets = []
            for sn in snippets:
                if len(sn) == 0:
                    print("Warning: empty snippets; ignoring")
                    auxtfidf = []
                else:
                    auxtfidf = self.tfidf.transform(sn)
                featuressnippets.append(auxtfidf)
            minsnippetdistances = [[0.5 if featuressnippets[i] == [] else np.amin(pairwise_distances(tfidfsnippets.getrow(i),
                                                                                                     featuressnippets[i],
                                                                                                     'cosine'))]
                                   for i in range(len(candidatesentences))]
            bestsnippets = [[0 if featuressnippets[i] == [] else 1 + np.argmin(pairwise_distances(tfidfsnippets.getrow(i),
                                                                                                  featuressnippets[i],
                                                                                                  'cosine'))]
                             for i in range(len(candidatesentences))]

        # embedding-based distance features
        print("extractfeatures [12345]\r", end='')
        sys.stdout.flush()
        alldistances = []
        for i in range(len(candidatesentences)):
            bar = "embedding distances ["+"="*int(i*10/len(candidatesentences))+" "*(10-int(i*10/len(candidatesentences)))+"] %i/%i" % (i,len(candidatesentences))
            print("%s\r" % (bar), end='')
            sys.stdout.flush()

            if candidatessentences_ids == []:
                features_preamble = []
            else:
                features_preamble = [candidatessentences_ids[i]]

            #stokens = list(set(my_tokenize(candidatesentences[i])))
            #qtokens = list(set(my_tokenize(questions[i])))

            # weighted embeddings
            stfidf = tfidfsnippets.getrow(i)
            qtfidf = tfidfquestions.getrow(i)

            sweights = []
            sembeddings = []
            for j in stfidf.nonzero()[1]:
                embedding = word2vec.vectors([tfidfvocabulary[j]])
                if len(embedding) == 0:
                    continue
                assert len(embedding) == 1
                sweights.append(stfidf[0,j])
                sembeddings.append(embedding[0])

            qweights = []
            qembeddings = []
            for j in qtfidf.nonzero()[1]:
                embedding = word2vec.vectors([tfidfvocabulary[j]])
                if len(embedding) == 0:
                    continue
                assert len(embedding) == 1
                qweights.append(qtfidf[0,j])
                qembeddings.append(embedding[0])

            if self.embedding_distances:
                # Cosine distance of sums of word embeddings
                if len(qembeddings) == 0 or len(sembeddings) == 0:
                    alldistances.append(features_preamble + [tfidfdistances[i][0],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    continue

                embeddistance = pairwise_distances([np.sum(qembeddings,0)],
                                                   [np.sum(sembeddings,0)],
                                                   'cosine')[0]

                # Pairwise distances
                pairdistances = pairwise_distances(qembeddings,sembeddings,'cosine')

                # Weighted pairwise similarities
                wpairsimilarities = []
                for qi in range(len(qweights)):
                    row = []
                    for si in range(len(sweights)):
                        row.append(qweights[qi]*sweights[si]*(1-pairdistances[qi,si]))
                    wpairsimilarities.append(row)
                wpairsimilarities = np.array(wpairsimilarities)
                sorteddistances = sorted(pairdistances.reshape(-1))
                sortedwsimilarities = sorted(wpairsimilarities.reshape(-1))

                distance = features_preamble + [tfidfdistances[i][0], embeddistance[0]]

                # Features using pairwise distances
                distance.append(np.mean(pairdistances)) # Mean
                distance.append(np.median(pairdistances)) # Median
                distance.append(np.amax(pairdistances)) # Maximum
                distance.append(np.mean(sorteddistances[-2:])) # Mean of 2 highest
                distance.append(np.mean(sorteddistances[-3:])) # Mean of 3 highest
                distance.append(np.amin(pairdistances)) # Minimum
                distance.append(np.mean(sorteddistances[:2])) # Mean of 2 lowest
                distance.append(np.mean(sorteddistances[:3])) # Mean of 3 lowest

                # Features using weighted pairwise similarities
                distance.append(np.mean(wpairsimilarities)) # Mean
                distance.append(np.median(wpairsimilarities)) # Median
                distance.append(np.amax(wpairsimilarities)) # Maximum
                distance.append(np.mean(sortedwsimilarities[-2:])) # Mean of 2 highest
                distance.append(np.mean(sortedwsimilarities[-3:])) # Mean of 3 highest
                distance.append(np.amin(wpairsimilarities)) # Minimum
                distance.append(np.mean(sortedwsimilarities[:2])) # Mean of 2 lowest
                distance.append(np.mean(sortedwsimilarities[:3])) # Mean of 3 lowest

                alldistances.append(distance)
            else:
                if len(qembeddings) == 0 or len(sembeddings) == 0:
                    embeddistance = [0.0]
                else:
                    embeddistance = pairwise_distances([np.sum(qembeddings,0)],
                                                       [np.sum(sembeddings,0)],
                                                       'cosine')[0]
                    #print("Embeddistance =", embeddistance)
                alldistances.append(features_preamble + [embeddistance[0]])

        print("                                                     \r",end='')
        sys.stdout.flush()

        if True: # self.embedding_distances:
            #print("Shape of alldistances:", np.array(alldistances).shape)
            #print("Shape of minsnippetdistances:", np.array(minsnippetdistances).shape)
            #print("Shape of bestsnippets:", np.array(bestsnippets).shape)
            #print("Shape of tfidfsnippets:", tfidfsnippets.shape)
            #print(alldistances[:5])
            sentencefeatures = scipy.sparse.hstack(
                (np.array(alldistances),
                 minsnippetdistances,
                 bestsnippets,
                 tfidfsnippets)
            ).tocsr()
        else:
            #print(alldistances)
            sentencefeatures = scipy.sparse.csr_matrix(alldistances)

        vectorembeddings = word2vec.Vectors(
            [my_tokenize(s) for s in candidatesentences]
        )
        return vectorembeddings, sentencefeatures


    # def extractfeatures(self,questions,candidatesentences):
    #     """Return the features - distance and word2vec
    #       questions - the questions
    #       candidate sentences - a list of sentences
    #     The length of the two lists must be equal"""

    #     assert len(questions) == len(candidatesentences)
    #     featuressnippets = [self.generate_sentence_embeddings(s) for s in candidatesentences]
    #     featuresquestions = [self.generate_sentence_embeddings(q) for q in questions]
    #     similarities = [pairwise_distances([featuresquestions[i]],
    #                                        [featuressnippets[i]],
    #                                        'cosine')[0]
    #                     for i in range(len(candidatesentences))]
    #     return scipy.hstack((similarities,featuressnippets))

    #def extractfeatures(self,questions,candidatesentences,snippets=[]):
    #    "Return the features as word embeddings"
    #    # return np.array([word2vec.vectors(my_tokenize(s)) for s in candidatesentences])
    #    return word2vec.Vectors([my_tokenize(s) for s in candidatesentences])

    def _collect_data(self, action, indices, testindices, loadpicklefile, savepicklefile, subsampling=0):
        "Collect the data for training or testing given the question indices"

        # print("Gathering data from EBM summariser")
        # # Collect text and labels of EBMSummariser
        # with open("multi_sents_rouge.csv") as f:
        #     rougeebmsum = dict()
        #     reader = csv.reader(f)
        #     header = next(reader)
        #     index = [header.index(m) for m in self.metric]
        #     for line in reader:
        #         key = (int(line[0]),int(line[1]),int(line[2]))
        #         rougeebmsum[key] = np.mean([float(line[i]) for i in index])

        # for (qid,sid,qtext,stext,sentences) in rouge_ebmsum.yieldSnipSentences("ClinicalInquiries.xml"):
        #     for abstract in sentences:
        #         candidatessentences += abstract
        #         candidatesquestions += [qtext]*len(abstract)
        #         allrouge += [rougeebmsum[(int(qid),int(sid),i)] for i in range(len(abstract))]

        candidatesquestions = [[], []]
        candidatessentences = [[], []]
        candidatessentences_ids = [[], []]
        snippetssentences = [[], []]
        allrouge = [[], []]

        if loadpicklefile:
            print("Loading features from pickle file %s" % loadpicklefile)
            with open(loadpicklefile, "br") as f:
                features, allrouge, self.tfidf = pickle.load(f)
        else:
            # Collect text and labels of trainset
            allquestions = [[], []]
            allanswers = [[], []]
            bins = [[[], [], [], [], [], [], [], [], [], []],
                    [[], [], [], [], [], [], [], [], [], []]]

            print("Gathering data from BioASQ data using %i train indices" % len(indices))

            for qi in range(len(self.data)):
                if qi in indices:
                    partition=0
                elif testindices != None and qi in testindices:
                    partition=1
                else:
                    continue

                allquestions[partition].append(self.data[qi]['body'])
                if type(self.data[qi]['ideal_answer']) == list:
                    allanswers[partition] += self.data[qi]['ideal_answer']
                else:
                    allanswers[partition].append(self.data[qi]['ideal_answer'])

                #ai = 0
                if 'snippets' not in self.data[qi].keys():
                    continue

                numcandidates = 0
                for (pubmedid,sentid,sent) in yield_candidate_text(self.data[qi]):
                # snippetssentences = []
                # for qsnipi in range(len(self.data[qi]['snippets'])):
                #     sentid = -1
                #     for sent in sent_tokenize(self.data[qi]['snippets'][qsnipi]['text']):
                #         sentid += 1
                        numcandidates += 1
                        candidatesquestions[partition].append(allquestions[partition][-1])
                        candidatessentences[partition].append(sent)
                        candidatessentences_ids[partition].append(sentid)
                        rougevalue = self.rouge[(qi,pubmedid,sentid)]
                        thisindex = len(allrouge[partition])
                        allrouge[partition].append(rougevalue)

                        #print(rougevalue)
                        #print(bins)
                        if rougevalue == 1.0:
                            bins[partition][9].append(thisindex)
                        else:
                            bins[partition][int(rougevalue*10)].append(thisindex)

    #                    allrouge.append(self.rouge[(qi,str(qsnipi),sentid)])

                if self.add_snippets:
                    datasnippetsentences = [s for sn in self.data[qi]['snippets']
                                            for s in sent_tokenize(sn['text'])]
                    if len(datasnippetsentences) == 0:
                        continue

                    snippetssentences[partition] += [datasnippetsentences] * numcandidates
                else:
                    snippetssentences[partition] += [[]] * numcandidates

            # Generate regression model
            print("Extracting features from gathered data")
            if action=="train":

                self.tfidf = TfidfVectorizer(tokenizer=my_tokenize)
                # self.tfidf.fit(allquestions+candidatessentences)
                # self.tfidf.fit(candidatessentences)
                self.tfidf.fit(allquestions[0]+allanswers[0]+candidatessentences[0])

                # Subsampling to keep balanced data
                if subsampling > 0:
                    print("Histogram of ROUGE before subsampling to %i samples per bin:" % subsampling)
                    candidatesq = []
                    candidatess = []
                    candidatessi = []
                    snippetss = []
                    rouge = []
                    for i in range(10):
                        print(len(bins[0][i]))
                        if len(bins[0][i]) == 0:
                            continue
                        cq = []
                        cs = []
                        csi = []
                        ss = []
                        r = []
                        for j in bins[0][i]:
                            cq.append(candidatesquestions[0][j])
                            cs.append(candidatessentences[0][j])
                            csi.append(candidatessentences_ids[0][j])
                            ss.append(snippetssentences[0][j])
                            r.append(allrouge[0][j])
                        #print(len(cq), len(cs), len(ss), len(r))
                        samples = np.random.choice(len(cq), subsampling)
                        for s in samples:
                            candidatesq.append(cq[s])
                            candidatess.append(cs[s])
                            candidatessi.append(csi[s])
                            snippetss.append(ss[s])
                            rouge.append(r[s])
                    candidatesquestions[0] = candidatesq
                    candidatessentences[0] = candidatess
                    candidatessentences_ids[0] = candidatessi
                    snippetssentences[0] = snippetss
                    allrouge[0] = rouge

            features = self.extractfeatures(candidatesquestions[0],
                                            candidatessentences[0],
                                            candidatessentences_ids[0],
                                            snippetssentences[0])

            if savepicklefile:
                print("Saving features in file %s" % (savepicklefile))
                with open(savepicklefile, 'bw') as f:
                    pickle.dump((features, allrouge, self.tfidf), f)
        return features, allrouge, candidatesquestions, candidatessentences, candidatessentences_ids, snippetssentences

    def save_for_hpc(self, savepath, corpusfile, rougefile, n_folds=10, small_data=False):
        "Save data for HPC"
        print("Saving HPC data in %s" % savepath)
        if small_data:
            indices = [i for i in range(len(self.data))][:20]
            subsampling = 10
        else:
            indices = [i for i in range(len(self.data))]
            subsampling = 1000

        random.seed(1234)
        random.shuffle(indices)
        fold = 0
        kf = KFold(n_splits=n_folds)
        for traini, testi in kf.split(indices):
            fold += 1
            if small_data and fold > 2:
                break

            print("Saving fold %i" % fold)
            trainindices = [indices[i] for i in traini]
            testindices = [indices[i] for i in testi]
            self._collect_data("train", trainindices, None, None,
                                "%s/%i%s" % (savepath, fold, "train.pickle"),
                               subsampling=subsampling)
            self._collect_data("test", testindices, None, None,
                                "%s/%i%s" % (savepath, fold, "test.pickle"))

    def run_for_hpc(self, loadpath=None, n_folds=10, fold=0, resultspath="", small_data=False):
        "Run models"
        def iterator_kfold(data, n_folds):
            "Iterator wrapper for KFold"
            kf = KFold(n_splits=n_folds)
            for traini, testi in kf.split(data):
                yield traini, testi

        subsampling = 0
        if loadpath is None:
            if small_data:
                indices = [i for i in range(len(self.data))][:20]
                subsampling = 10
            else:
                indices = [i for i in range(len(self.data))]
                subsampling = 1000

            random.seed(1234)
            random.shuffle(indices)
            kfolds = iterator_kfold(indices, n_folds)

        results = []
        for fold_i in range(1, n_folds + 1):
            if small_data and fold_i > 2:
                break
            if fold > 0 and fold_i != fold:
                continue
            print("Running fold %i" % fold_i)
            if loadpath is None:
                train_loadpath = None
                test_loadpath = None
                traini, testi = next(kfolds)
                trainindices = [indices[i] for i in traini]
                testindices = [indices[i] for i in testi]
            else:
                train_loadpath = "%s/%i%s" % (loadpath, fold_i, "train.pickle")
                test_loadpath = "%s/%i%s" % (loadpath, fold_i, "test.pickle")
                trainindices = None
                testindices = None

            features, allrouge, candidatesquestions, candidatessentences, snippetssentences = \
                self._collect_data("train", trainindices, None, train_loadpath, None, subsampling=subsampling)

            # print(features.shape())
            if self.gamma == 0:
                self.regression = svm.SVR(kernel=self.kernel,
                                          C=self.C,
                                          degree=self.degree)
            else:
                self.regression = svm.SVR(kernel=self.kernel,
                                          C=self.C,
                                          degree=self.degree,
                                          gamma=self.gamma)


            print("Training SVR")
            self.regression.fit(features, allrouge[0])
            predictions = self.regression.predict(features)
            trainloss = np.mean((predictions - np.array(allrouge[0])) ** 2)

            features, allrouge, candidatesquestions, candidatessentences, snippetssentences = \
                self._collect_data("test", testindices, None, test_loadpath, None)

            print("Testing SVR")
            predictions = self.regression.predict(features)
            if resultspath != "":
                print("Saving predictions")
                with open("%s/testresults_%i.csv" % (resultspath, fold_i), "w") as f:
                    writer = csv.DictWriter(f, fieldnames=["id", "target", "prediction"])
                    writer.writeheader()
                    for i, p in enumerate(predictions):
                        writer.writerow({"id": i,
                                         "target": allrouge[0][i],
                                         "prediction": p})
                print("Saving model")
                with open("%s/model_%i.pickle" % (resultspath, fold_i), "wb") as f:
                    pickle.dump(self.regression, f)

            testloss = np.mean((predictions - np.array(allrouge[0])) ** 2)

            print("TrainMSE: %1.5f TestMSE: %1.5f" % (trainloss, testloss))
            results.append((trainloss, testloss))

        if fold == 0:
            print("%5s %7s %7s" % ('', 'TrainMSE', 'TestMSE'))
            for i, r in enumerate(results):
                print("%5i %1.5f %1.5f" % (i + 1, r[0], r[1]))
            mean_Trainloss = np.average([r[0] for r in results])
            mean_Testloss = np.average([r[1] for r in results])
            print("%5s %1.5f %1.5f" % ("mean", mean_Trainloss, mean_Testloss))
            std_Trainloss = np.std([r[0] for r in results])
            std_Testloss = np.std([r[1] for r in results])
            print("%5s %1.5f %1.5f" % ("stdev", std_Trainloss, std_Testloss))
            return mean_Trainloss, std_Trainloss, mean_Testloss, std_Testloss
        else:
            print("%5s %7s %7s" % ('Fold', 'TrainMSE', 'TestMSE'))
            print("%5i %1.5f %1.5f" % (fold, results[0][0], results[0][1]))
            return results[0][0], 0, results[0][1], 0


class Regression(BaseRegression):
    "A regression system that uses SVR"
    def __init__(self, corpusFile, rougeFile, add_snippets=False,
                 embedding_distances=True,
                 metric=["SU4"],
                 kernel="rbf", C=1.0, gamma=0, degree=3):
        """Initialise the regression system.

        metric is a list of possible ROUGE metrics. If there are more
        than one, the target metric is the average of the metrics
        listed.
        """
        BaseRegression.__init__(self, corpusFile, rougeFile,
                                add_snippets=add_snippets,
                                embedding_distances=embedding_distances,
                                metric=metric)
        self.metric = metric
        self.kernel = kernel
        self.degree = degree
        if kernel == 'poly5':
            self.kernel = 'poly'
            self.degree = 5
        elif kernel == 'poly7':
            self.kernel = 'poly'
            self.degree = 7
        self.gamma = gamma
        self.C = C

    def extractfeatures(self, questions, candidatesentences, candidatessentences_ids=[], snippets=[]):
        """Return the features - based on distance metrics between question and text
          questions - the questions
          candidatesentences - a list of sentences
          snippets - a list of retrieved snippets if available; each element is a tuple
               (numberofcandidates,snippetsofthesecandidates)
        The length of the two lists must be equal"""

        embeddings, features = BaseRegression.extractfeatures(self,
                                                              questions,
                                                              candidatesentences,
                                                              candidatessentences_ids,
                                                              snippets)
        vector_embeddings = []
        for e in embeddings:
            if len(e) == 0:
                vector_embeddings.append(np.zeros(word2vec.DIMENSION))
            else:
                vector_embeddings.append(np.mean(e,0))

        return scipy.sparse.hstack((vector_embeddings, features)).tocsr()


    def train(self,indices,testindices=None,loadpicklefile=None,savepicklefile=None):
        "Train the regressor given the question indices"
        features, allrouge, candidatesquestions, candidatessentences, candidatessentences_ids, snippetssentences = \
        BaseRegression._collect_data(self, "train", indices, testindices,
                                     loadpicklefile, savepicklefile)
        if self.embedding_distances:
            str_embedding = "with"
        else:
            str_embedding = "without"
        print("Training SVR %s embedding distances" % (str_embedding))
        if self.gamma == 0:
            self.regression = svm.SVR(kernel=self.kernel,
                                      C=self.C,
                                      degree=self.degree)
        else:
            self.regression = svm.SVR(kernel=self.kernel,
                                      C=self.C,
                                      gamma=self.gamma,
                                      degree=self.degree)
        self.regression.fit(features,allrouge[0])
        predictions = self.regression.predict(features)
        losstrain = np.mean((predictions-np.array(allrouge[0]))**2)

        if testindices == None:
            return losstrain
        else:
            featurestest = self.extractfeatures(candidatesquestions[1],
                                                candidatessentences[1],
                                                candidatessentences_ids[1],
                                                snippetssentences[1])
            predictionstest = self.regression.predict(featurestest)
            losstest = np.mean((predictionstest-np.array(allrouge[1]))**2)
            return losstrain, losstest

    def test(self,indices,loadpicklefile=None,savepicklefile=None):
        "Test the regressor given the question indices"
        features, allrouge, candidatesquestions, candidatessentences, candidatessentences_ids, snippetssentences = \
        BaseRegression._collect_data(self, "test", indices, None, loadpicklefile, savepicklefile)
        print("Testing SVR")
        predictions = self.regression.predict(features)
        loss =  np.mean((predictions-np.array(allrouge[0]))**2)
        print("MSE = %f" % loss)
        return loss

    def answersummary(self, question, candidatesentences_and_ids, snippetsentences,
                      n=3, qindex=None):
        """Return a summary that answers the question

        qindex is the question index, it is used for caching features"""

        # TODO: Cache features for answersummary

        # snippetsentences = [s for snippet in snippets
        #                       for s in sent_tokenize(snippet)]
        candidatesquestions = [question]*len(candidatesentences_and_ids)
        candidatesentences, sentence_ids = zip(*candidatesentences_and_ids)
        features = self.extractfeatures(candidatesquestions, candidatesentences,
                                        sentence_ids,
                                        [snippetsentences]*len(candidatesentences))
        predictions = self.regression.predict(features)
        # print("Predictions:",predictions)
        scores = list(zip(predictions,range(len(candidatesentences))))
        scores.sort()
        summary = scores[-n:]
        summary.sort(key = lambda x: x[1])
        return [candidatesentences[i] for (score,i) in summary]

class RegressionWithoutPos(Regression):
    "A regression system that uses SVR"
    def __init__(self, corpusFile, rougeFile, add_snippets=False,
                 embedding_distances=True,
                 metric=["SU4"],
                 kernel="rbf", C=1.0, gamma=0, degree=3):
        """Initialise the regression system.

        metric is a list of possible ROUGE metrics. If there are more
        than one, the target metric is the average of the metrics
        listed.
        """
        Regression.__init__(self, corpusFile, rougeFile,
                            add_snippets=add_snippets,
                            embedding_distances=embedding_distances,
                            metric=metric, kernel=kernel,
                            C=C, gamma=gamma, degree=degree)
        print("Starting regression without position information")

    def extractfeatures(self, questions, candidatesentences, candidatessentences_ids=[], snippets=[]):
        """Return the features - based on distance metrics between question and text
          questions - the questions
          candidatesentences - a list of sentences
          snippets - a list of retrieved snippets if available; each element is a tuple
               (numberofcandidates,snippetsofthesecandidates)
        The length of the two lists must be equal"""

        return Regression.extractfeatures(self,
                                          questions,
                                          candidatesentences,
                                          candidatessentences_ids=[],
                                          snippets=snippets)

class OracleRegression:
    "Oracle for regression"
    def __init__(self,corpusFile,rougeFile,metric=["SU4"]):
        "Initialisation"
        print("Reading data from %s and %s" % (corpusFile,rougeFile))
        self.metric = metric
        self.data = loaddata(corpusFile)
        self.rouge = dict()
        with open(rougeFile) as f:
            reader = csv.reader(f)
            header = next(reader)
            index = [header.index(m) for m in metric]
            for line in reader:
#                key = (int(line[0]),int(line[1]),int(line[2]))
                key = (int(line[0]),line[1],int(line[2]))
                self.rouge[key] = np.mean([float(line[i]) for i in index])

    def train(self,trainindices,testindices):
        "Training the oracle"
        allsentences = []
        for qi in range(len(self.data)):
            if qi in trainindices:
                allsentences += [s for (pid,sid,s) in yield_candidate_text(self.data[qi])]
        self.tfidf = TfidfVectorizer(tokenizer=my_tokenize)
        self.tfidf.fit(allsentences)
        return 0.0,0.0

    def answersummary(self,_question,_candidatesentences,_snippetsentences,n,qindex,threshold=0.3):
        """Return a sumary based on the known ROUGE of the individual sentences
        The algorithm will select the top n sentences in a greedy procedure so that
        the distance of a new sentence to the other sentences is higher than the threshold."""
        candidatesentences = []
        rougevalues = []
        for (pubmedid,senti,sent) in yield_candidate_text(self.data[qindex]):
        # for qsnipi in range(len(self.data[qindex]['snippets'])):
        #     senti = -1
        #     for sent in sent_tokenize(self.data[qindex]['snippets'][qsnipi]['text']):
        #         senti += 1
                candidatesentences.append(sent)
#                rougevalues.append(self.rouge[(qindex,qsnipi,senti)])
                rougevalues.append(self.rouge[(qindex,pubmedid,senti)])

        scores = list(zip(rougevalues,range(len(candidatesentences))))
        scores.sort(reverse=True)
        tfidf = self.tfidf.transform([candidatesentences[si] for (rouge,si) in scores])
        distances = pairwise_distances(tfidf,metric='cosine')
        summary = []
        for (rouge,si) in scores:
            if len(summary) >= n:
                break
            summarydistances = [distances[si,sj] for sj in summary if si!=sj]
            if len(summarydistances) == 0:
                summary.append(si)
            elif np.min(summarydistances) >= threshold:
                summary.append(si)
        summary.sort()
        return [candidatesentences[i] for i in summary]

def evaluate(regressionClassInstance, rougeFilename="rouge.xml",
             nanswers = {"summary": 6,
                         "factoid": 2,
                         "yesno": 2,
                         "list": 3},
             add_snippets=False, small_data=False, loadpath=None, fold=0, resultspath="."):
    """Evaluate a regression-based summariser

    nanswers is the number of answers. If it is a dictionary, then the keys indicate the question type, e.g.
    nanswers = {"summary": 6,
                "factoid": 2,
                "yesno": 2,
                "list": 3}
"""
    dataset = regressionClassInstance.data
    #indices = range(len(dataset))
    indices = [i for i in range(len(dataset))
               #if dataset[i]['type'] == 'summary'
               #if dataset[i]['type'] == 'factoid'
               #if dataset[i]['type'] == 'yesno'
               #if dataset[i]['type'] == 'list'
               ]
    if small_data:
        indices = indices[:20]

    random.seed(1234)
    random.shuffle(indices)

    rouge_results = []
    fold_i = 0
    kf = KFold(n_splits=10)
    for (traini, testi) in kf.split(indices):
        fold_i += 1
        if fold != 0 and fold_i != fold:
            continue

        if fold == 0 and small_data and fold_i > 2:
            break

        print("Cross-validation Fold %i" % fold_i)
        trainindices = [indices[i] for i in traini]
        testindices = [indices[i] for i in testi]

#        ident = "embeddings"
#        trainloss = regressionClassInstance.train(trainindices,
#                                                  savepicklefile="trainfeatures_nn%s_%i.pickle" % (ident, fold))
#        testloss = regressionClassInstance.test(testindices,
#                                                savepicklefile="testfeatures_nn%s_%i.pickle" % (ident,fold))
#        trainloss = regressionClassInstance.train(trainindices,
#                                                  loadpicklefile="trainfeatures_nn%s_%i.pickle" % (ident,fold))
#        testloss = regressionClassInstance.test(testindices,
#                                                loadpicklefile="testfeatures_nn%s_%i.pickle" % (ident,fold))
        if loadpath is None:
            (trainloss, testloss) = regressionClassInstance.train(trainindices, testindices)
        else:
            train_loadpath = "%s/%i%s" % (loadpath, fold_i, "train.pickle")
            trainloss = regressionClassInstance.train(trainindices, None,
                                                      loadpicklefile=train_loadpath)
            test_loadpath = "%s/%i%s" % (loadpath, fold_i, "test.pickle")
            testloss = regressionClassInstance.test(testindices, loadpicklefile=test_loadpath)
        eval_test_target = []
        eval_test_system = []
        with open("%s/%s" % (resultspath, rougeFilename), 'w') as frouge:
            print("Collecting evaluation results")
            frouge.write('<ROUGE-EVAL version="1.0">\n')
            for di in range(len(dataset)):
                if di not in testindices:
                    continue
                question = dataset[di]['body']
                if 'snippets' not in dataset[di].keys():
                    continue
                candidates = [(sent, sentid)
                              for (pubmedid, sentid, sent) in yield_candidate_text(dataset[di])]
                if add_snippets:
                    snippetsentences = [s for snippet in dataset[di]['snippets']
                                        for s in sent_tokenize(snippet['text'])]
                else:
                    snippetsentences = []
                #print("Snippet sentences:", snippetsentences)
                # candidates = [s for snippet in dataset[di]['snippets']
                #               for s in sent_tokenize(snippet['text'])]
                # snippetsentences = candidates
                if len(candidates) == 0:
                    # print("Warning: No text to summarise; ignoring this text")
                    continue

                if type(nanswers) == dict:
                    n = nanswers[dataset[di]['type']]
                else:
                    n = nanswers
                summary = regressionClassInstance.answersummary(question,
                                                                candidates,
                                                                snippetsentences,
                                                                n=n,
                                                                qindex=di)
                frouge.write("""<EVAL ID="%i">
 <MODEL-ROOT>
 %s/rouge/models
 </MODEL-ROOT>
 <PEER-ROOT>
 %s/rouge/summaries
 </PEER-ROOT>
 <INPUT-FORMAT TYPE="SEE">
 </INPUT-FORMAT>
""" % (di, resultspath, resultspath))
                frouge.write(""" <PEERS>
  <P ID="A">summary%i.txt</P>
 </PEERS>
 <MODELS>
""" % (di))
                with codecs.open(os.path.join(resultspath,'rouge','summaries',
                                              'summary%i.txt' % (di)),
                             'w', 'utf-8') as fout:
                    a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</a></body>\n</html>'.format(" ".join(summary))
                    fout.write(a+'\n')
                    # fout.write('\n'.join([s for s in summary])+'\n')

                if type(dataset[di]['ideal_answer']) == list:
                    ideal_answers = dataset[di]['ideal_answer']
                else:
                    ideal_answers = [dataset[di]['ideal_answer']]

                for j in range(len(ideal_answers)):
                    frouge.write('  <M ID="%i">ideal_answer%i_%i.txt</M>\n' % (j,di,j))
                    with codecs.open(os.path.join(resultspath,'rouge','models',
                                                  'ideal_answer%i_%i.txt' % (di,j)),
                                     'w', 'utf-8') as fout:
                        a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</a></body>\n</html>'.format(ideal_answers[j])
                        # a='\n'.join(sent_tokenize(ideal_answers[j]))
                        fout.write(a+'\n')
                frouge.write(""" </MODELS>
</EVAL>
""")
                eval_test_system.append({'id': dataset[di]['id'],
                                         'ideal_answer': " ".join(summary),
                                         'exact_answer': ""})
                eval_test_target.append({'id': dataset[di]['id'],
                                         'ideal_answer': ideal_answers,
                                         'exact_answer': ""})

            frouge.write('</ROUGE-EVAL>\n')

        json_summaries_file = "%s/crossvalidation/crossvalidation_%i_summaries.json" % (resultspath, fold_i)
        print("Saving summaries in file %s" % json_summaries_file)
        with open(json_summaries_file,'w') as fcv:
            fcv.write(json.dumps({'questions': eval_test_system}, indent=2))
        json_gold_file = "%s/crossvalidation/crossvalidation_%i_gold.json" % (resultspath, fold_i)
        print("Saving gold data in in file %s" % json_gold_file)
        with open(json_gold_file,'w') as fcv:
            fcv.write(json.dumps({'questions': eval_test_target}, indent=2))


        print("Calling ROUGE", rougeFilename)
        ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ../rouge/RELEASE-1.5.5/data -c 95 -2 4 -u -x -n 4 -a ' + "%s/%s" % (resultspath, rougeFilename)
        # ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ../rouge/RELEASE-1.5.5/data -a -n 2 -2 4 -U '+rougeFilename
        #os.system(ROUGE_CMD)
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
        rouge_results.append(F)


        ### This code generates JSON test data for double checks; it can be deleted ####
        #bioasq_test_file = 'phaseB_3b_05.json'
        #print("Using BioASQ test file %s" % bioasq_test_file)
        #bioasq_testset = loaddata(bioasq_test_file)
        #bioasq_answers = yield_bioasq_answers(regressionClassInstance,
        #                                      bioasq_testset,
        #                                      nanswers=nanswers)
        #bioasq_result = {"questions": [a for a in bioasq_answers]}
        #bioasq_output_file = 'crossvalidation/bioasq_3b_05_%i.json' % fold_i
        #print("Saving BioASQ results in file %s" % bioasq_output_file)
        #with open(bioasq_output_file, 'w') as fcv:
        #    fcv.write(json.dumps(bioasq_result, indent=2))
        ### End of additional code ###

    print("%5s %7s %7s %7s %7s" % ('', 'N-2', 'SU4', 'TrainMSE', 'TestMSE'))
    for i in range(len(rouge_results)):
        print("%5i %1.5f %1.5f %1.5f %1.5f" % (i+1,rouge_results[i]['N-2'],rouge_results[i]['SU4'],
                                       rouge_results[i]['trainloss'],rouge_results[i]['testloss']))
    mean_N2 = np.average([rouge_results[i]['N-2']
                          for i in range(len(rouge_results))])
    mean_SU4 = np.average([rouge_results[i]['SU4']
                           for i in range(len(rouge_results))])
    mean_Trainloss = np.average([rouge_results[i]['trainloss']
                                 for i in range(len(rouge_results))])
    mean_Testloss = np.average([rouge_results[i]['testloss']
                                for i in range(len(rouge_results))])
    print("%5s %1.5f %1.5f %1.5f %1.5f" % ("mean",mean_N2,mean_SU4,mean_Trainloss,mean_Testloss))
    stdev_N2 = np.std([rouge_results[i]['N-2']
                       for i in range(len(rouge_results))])
    stdev_SU4 = np.std([rouge_results[i]['SU4']
                        for i in range(len(rouge_results))])
    stdev_Trainloss = np.std([rouge_results[i]['trainloss']
                              for i in range(len(rouge_results))])
    stdev_Testloss = np.std([rouge_results[i]['testloss']
                             for i in range(len(rouge_results))])
    print("%5s %1.5f %1.5f %1.5f %1.5f" % ("stdev",stdev_N2,stdev_SU4,stdev_Trainloss,stdev_Testloss))
    print()
    return mean_SU4, stdev_SU4



if __name__ == "__main__":
    import doctest
    doctest.testmod()
    #saveRouge('BioASQ-training7b.json','rouge_7b.csv')
    #import sys
    #sys.exit()

# r = Regression.load("task6b_svr.pickle")
# bioasq(r)

    # import sys
#     # if len(sys.argv) == 1:
#     #     gridregression()
#     # else:
#     #     gridregression(idx=int(sys.argv[1]))

    # sys.exit()

    import argparse
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--hidden",
                        type=int,
                        default=50,
                        help="Size of hidden layer")
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=3,
                        help="Number of epochs")
    parser.add_argument("-r", "--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout rate")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=2,
                        help="Verbosity")
    parser.add_argument("-n", "--ngrams", nargs='+',
                        type=int,
                        default=[2,3,4],
                        help="Ngrams for convolutions")
    parser.add_argument('-t', '--regression_type',
                        choices=("SVR"), #, "Oracle"),
                        default="SVR")
    parser.add_argument('-s', '--add_snippets', action="store_true")
    parser.add_argument('-m', '--embedding_distances', action="store_true")

    parser.add_argument("-k", "--kernel",
                        choices=["rbf","poly","sigmoid"],
                        default="rbf",
                        help="Type of kernel")
    parser.add_argument("-C", "--C", type=float, default=1.0,
                        help="Penalty parameter of the error term")
    parser.add_argument("-g", "--gamma", type=float, default=0.1,
                        help="Kernel coefficient")
    parser.add_argument("-D", "--degree", type=int, default=5,
                        help="Polynomial degree")
    parser.add_argument("-S", "--small_data", action="store_true",
                        help="Use small data for debugging and testing")
    parser.add_argument("-f", "--fold", type=int, default=0,
                        help="Use only the specified fold (0 for all folds)")
    parser.add_argument("-p", "--tmp_path", default=".",
                        help="Path for temporary files")
    args = parser.parse_args()

    if args.small_data:
        print("Using small data for debugging")

    if args.add_snippets:
        print("Including distance to closest snippet")
    else:
        print("NOT including distance to closest snippet")

    if args.embedding_distances:
        print("Including embedding distances")
    else:
        print("NOT included embedding distances")

    if args.regression_type == "SVR":
        if args.gamma == 0:
            print("Using SVR with kernel=%s and C=%f" % (args.kernel,
                                                         args.C))
        else:
            print("Using SVR with kernel=%s, C=%f and gamma=%f" % (args.kernel,
                                                                   args.C,
                                                                   args.gamma))

          #    saveRouge('BioASQ-trainingDataset4b.json','rouge_4b.csv')
      #   saveRouge('BioASQ-trainingDataset4b.json','rouge_4bPubMed.csv')
      #   saveRouge('BioASQ-trainingDataset2b.json','rouge_2b.csv')

        # r = Regression('BioASQ-trainingDataset4b.json', 'rouge_4bPubMed.csv',
        #                savecsvfile = 'regressionfeatures.csv')
        # allindices = range(len(r.data))[:10]
        # r.train(allindices)

        #import sys
        #sys.exit()
        regressor = Regression(#'golden_13.json',
                               #'train3b_2_5_from_batches.json',
                               #'BioASQ-trainingDataset6b.json',
                               'BioASQ-training7b.json',
                               #'golden_13PubMed.csv',
                               #'train3b_2_5_from_batchesPubMed.csv',
                               #'rouge_6bPubMed.csv',
                               #'rouge_6b.csv',
                               'rouge_7b.csv',
                               kernel=args.kernel,
                               C=args.C,
                               gamma=args.gamma,
                               degree=args.degree,
                               add_snippets=args.add_snippets,
                               embedding_distances=args.embedding_distances)

    #import sys
    #indices = list(range(len(regressor.data)))
    #if args.small_data:
    #    indices = indices[:20]
    #regressor.train(indices)
    #regressor.save("task6b_svr.pickle")
    #sys.exit()

    #import sys
    #regressor.save_for_hpc("hpcdata", 'BioASQ-trainingDataset6b.json', 'rouge_6bPubMed.csv', small_data=args.small_data)
    #sys.exit()


#    if args.regression_type == "Oracle":
#        evaluate(OracleRegression('BioASQ-trainingDataset4b.json',
#                                  'rouge_4b.csv'))
#    else:

    #mean_train_SU4, stdev_train_SU4, mean_SU4, stdev_SU4 = regressor.run_for_hpc(# loadpath="hpcdata",
    #                                                                             fold=args.fold,
    #                                                                             resultspath=args.tmp_path+"/crossvalidation",
    #                                                                             small_data=args.small_data)

    mean_SU4, stdev_SU4 = evaluate(regressor,
                                   nanswers={"summary": 6,
                                             "factoid": 2,
                                             "yesno": 2,
                                             "list": 3},
                                   #loadpath="hpcdata",
                                   fold=args.fold,
                                   resultspath=args.tmp_path,
                                   small_data=args.small_data,
                                   add_snippets=args.add_snippets)
    end_time = time.time()
    print("Time elapsed: %s" %
          (time.strftime("%X",
                         time.gmtime(end_time - start_time))))

    if args.regression_type == "SVR":
        print("|Fold|Method|kernel|C|gamma|degree|meanSU4|stdevSU4|")
        if args.gamma == 0:
            print("|%i|SVR|%s|%f|default|%i|%f|%f|" % (args.fold,
                                                       args.kernel, args.C, args.degree,
                                                       mean_SU4, stdev_SU4))
        else:
            print("|%i|SVR|%s|%f|%f|%i|%f|%f|" % (args.fold,
                                                  args.kernel, args.C, args.gamma,
                                                  args.degree,
                                                  mean_SU4, stdev_SU4))
