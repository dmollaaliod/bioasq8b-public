import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import nltk
porter = nltk.PorterStemmer()
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

import os.path

#import clinical
#from ilp3 import topilp, topnbrute
import word2vec

from xml_abstract_retriever import getAbstract

def loaddata(filename, qtype=None):
    """Load the JSON data
    >>> data = loaddata('BioASQ-training8b.json')
    >>> len(data)
    3243
    >>> sorted(data[0].keys())
    ['body', 'concepts', 'documents', 'id', 'ideal_answer', 'snippets', 'type']
    """
    data = json.load(open(filename))
    questions = [x for x in data['questions'] if 'ideal_answer' in x]
    if not qtype:
        return questions
    return [x for x in questions if x['type'] == qtype]

def load_test_data(filename):
    """Load the JSON data
    >>> data = load_test_data('BioASQ-training8b.json')
    >>> len(data)
    3243
    >>> sorted(data[0].keys())
    ['body', 'concepts', 'documents', 'id', 'ideal_answer', 'snippets', 'type']
    """
    print("Loading", filename)
    data = json.load(open(filename, encoding="utf-8"))
    return data['questions']

def tokenize(string,stem=True):
    """Tokenise a string
    >>> tokenize("This is a sentence. This is another sentence.")
    ['sentenc', 'anoth', 'sentenc']
    """
    if stem:
        return [porter.stem(w.lower())
                for s in sent_tokenize(string)
                for w in word_tokenize(s)
                if w.lower() not in stopwords.words('english') and
                w not in [',','.',';','(',')','"',"'",'=',':','%','[',']']]
    else:
        return [w.lower()
                for s in sent_tokenize(string)
                for w in word_tokenize(s)
                if w.lower() not in stopwords.words('english') and
                w not in [',','.',';','(',')','"',"'",'=',':','%','[',']']]

def jaccard_similarity(x,y):
    return float(len(x&y))/len(x|y)

def yield_text(dataset):
    """Yield text ready for fitting tf.idf
    >>> data = loaddata('BioASQ-training8b.json')
    >>> g = yield_text(data)
    >>> next(g)
    'Is Hirschsprung disease a mendelian or a multifactorial disorder?'
    >>> next(g)
    "Coding sequence mutations in RET, GDNF, EDNRB, EDN3, and SOX10 are involved in the development of Hirschsprung disease. The majority of these genes was shown to be related to Mendelian syndromic forms of Hirschsprung's disease, whereas the non-Mendelian inheritance of sporadic non-syndromic Hirschsprung disease proved to be complex; involvement of multiple loci was demonstrated in a multiplicative model."
"""
    for r in dataset:
        yield r['body']
        if type(r['ideal_answer']) == list:
            ideal_answers = r['ideal_answer']
        else:
            ideal_answers = [r['ideal_answer']]
        for s in ideal_answers:
            yield s
        # yield r['body']+" "+' '.join(ideal_answers)
        # for s in r['snippets']:
        #     yield s['text']

    # # Additional training data
    # for r in clinical.clinical_data.keys():
    #     record = clinical.clinical_data[r]
    #     qstring = record['q']
    #     for ak in record['a'].keys():
    #         yield qstring+" "+record['a'][ak]

# def yield_candidate_text(questiondata):
#     """Yield all candidate text for a question
#     >>> data = loaddata("BioASQ-trainingDataset6b.json")
#     >>> y = yield_candidate_text(data[0])
#     >>> next(y)
#     'The identification of common variants that contribute to the genesis of human inherited disorders remains a significant challenge.'
#     >>> next(y)
#     'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes.'
# """
#     for sn in questiondata['snippets']:
#         pubmedid = os.path.basename(sn['document'])
#         filename = os.path.join("Task5bPubMed",pubmedid+".xml")
#         for s in sent_tokenize(getAbstract(filename,version="0")[0]):
#             yield s

def yield_candidate_snippets(questiondata):
    """Yield all candidate text for a question
    >>> data = loaddata("BioASQ-training8b.json")
    >>> y = yield_candidate_snippets(data[0])
    >>> next(y)
    'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes'
    >>> next(y)
    "In this study, we review the identification of genes and loci involved in the non-syndromic common form and syndromic Mendelian forms of Hirschsprung's disease."
"""
    if 'snippets' not in questiondata:
        print("Warning: question without snippets:")
        print(questiondata['body'])
        return
    for sn in questiondata['snippets']:
        for s in sent_tokenize(sn['text']):
            yield s

def topn(scores,n):
    """Return the positions of the top n scores, in order of importance
    >>> topn((0.2, 2.0, 1.4, 0.3, 4.3), 3)
    [4, 1, 2]"""
#    return range(len(scores))
    data = list(zip(scores,range(len(scores))))
    data.sort(reverse=True)
#    print([x[1] for x in data[:n]])
    return [x[1] for x in data[:n]]

def topmmr(q_similarity,t_similarity,lmb,n):
    "Return the positions of the top n scores using Carbonell & Goldstein's MMR"
    #print(q_similarity)
    #print(t_similarity)
    allids = range(len(q_similarity))
    result = [q_similarity.argmax()]
    allids.remove(result[-1])
#    print(q_similarity.T)
    for i in range(1,min(len(q_similarity),n)):
        scores = [lmb*q_similarity[t]-(1-lmb)*max([t_similarity[t,t2]
                                                   for t2 in result])
                  for t in allids]
#        print(np.array(scores).T)
#        if np.max(scores) < 0:
#            break
        newi = np.argmax(scores)
        result.append(allids[newi])
        allids.remove(result[-1])
#    print(result)
#    print()
    return result

def my_normalize(array):
    X = array # No normalisation
#    X = (array-array.min())/(array.max()-array.min())
#    X = (array-array.min(0,keepdims=True))/(array.max(0,keepdims=True)-array.min(0,keepdims=True))
#    X = (array-array.min(1,keepdims=True))/(array.max(1,keepdims=True)-array.min(1,keepdims=True))
#    X[np.isnan(X)] = 0.5
    return X

def reduce_embeddings(embeddings,dimension=200):
    """Reduce the embeddings of a list of words
    >>> reduce_embeddings([[2.,3.,4.],[1.,2.,1.]],dimension=3)
    array([3., 5., 5.])
    >>> reduce_embeddings([],dimension=3)
    array([0., 0., 0.])
"""
    if len(embeddings) == 0:
        return np.zeros(dimension)
    return np.sum(embeddings,0)


def yield_answers(test,tfidf,lsa,nanswers=3):
    """Yield the answer of each record.

The parameter nanswers determines the number of selected sentences. If
it is a dictionary, then it indicates the number of selected sentences
for each type of question, e.g.
  nanswers = {"summary": 7,
              "factoid": 2,
              "yesno": 2,
              "list": 3}
"""
    questions = [r['body'] for r in test]
    q_embedding = np.array([reduce_embeddings(word2vec.vectors(tokenize(q,stem=False)),
                                              dimension=word2vec.DIMENSION)
                   for q in questions])
    q_embedding = my_normalize(q_embedding)

    q_tfidf = tfidf.transform(questions)
    q_lsa = lsa.transform(q_tfidf)

    text = [t for r in test for t in yield_candidate_snippets(r)]

    t_embedding = np.array([reduce_embeddings(word2vec.vectors(tokenize(t,stem=False)),
                                              dimension=word2vec.DIMENSION)
                   for t in text])
    t_embedding = my_normalize(t_embedding)
    print("Shape of the embeddings of the test data:", t_embedding.shape)
    # print(t_embedding.shape, q_embedding.shape)
    # print(t_embedding[0].shape)

    t_tfidf = tfidf.transform(text)
    # print("Min TFIDF:",t_tfidf.toarray().min(), "Max TFIDF:", t_tfidf.toarray().max())
    t_lsa = lsa.transform(t_tfidf)
    # print("Min LSA:",t_lsa.min(), "Max LSA:", t_lsa.max())

    #vocabulary = tfidf.get_feature_names()
    iq = 0
    it = 0
    itemnumber = 0
    for ir, r in enumerate(test):
        percentage_done = int(ir * 50 / len(test))
        print("[" + "=" * percentage_done + " " * (50 - percentage_done) + "]\r", end='')
        sys.stdout.flush()

        itemnumber += 1
        startit = it
        #snippets = r['snippets']
        snippets = [t for t in yield_candidate_snippets(r)]
        it += len(snippets)

        if type(nanswers) == dict:
            n = nanswers[r['type']]
        else:
            n = nanswers

        w_cos_q_lsa_t_lsa = np.array([cosine_similarity([x],[q_lsa[iq,:]])
                                      for x in t_lsa[startit:it,:]])
        w_cos_embeddings = np.array([cosine_similarity([x],[q_embedding[iq,:]])
                                      for x in t_embedding[startit:it]])

        iq += 1
        randomIDs = list(range(it-startit))
        random.shuffle(randomIDs)
        if len(snippets) > 0:
            w_cos_q_lsa_t_lsa = w_cos_q_lsa_t_lsa[:,0,0]
            w_cos_embeddings = w_cos_embeddings[:,0,0]
        else:
            pass
            #print("Warning: No snippets found in this record:")
            #print(r)

        #print("%i/%i; text size: %i" % (itemnumber,len(test),
        #                                len(w_cos_q_lsa_t_lsa)))

        if ir == len(test) - 1:
            print("[" + "=" * 50 + "]")

        yield [snippets[:n],
               [snippets[x] for x in randomIDs[:n]],
               [snippets[x] for x in topn(w_cos_q_lsa_t_lsa,n)],
               [snippets[x] for x in topn(w_cos_embeddings,n)]]

def bioasq_baseline(test_data='phaseB_5b_01.json',
                    output_file='bioasq-out-baseline.json'):
    """A simple baseline for BioASQ"""
    print("Processing baseline")
    nanswers = {"summary": 6,
                "factoid": 2,
                "yesno": 2,
                "list": 3}
    testset = load_test_data(test_data)
    results = []
    for r in testset:
        snippets = [s for s in yield_candidate_snippets(r)]
        if type(nanswers) == dict:
            n = nanswers[r['type']]
        else:
            n = nanswers
        if r['type'] == "yesno":
            exactanswer = "yes"
        else:
            exactanswer = ""
        results.append({"id": r['id'],
                        "ideal_answer": " ".join(snippets[:n]),
                        "exact_answer": exactanswer})
    print("Saving results in file %s" % output_file)
    with open(output_file, 'w') as f:
        f.write(json.dumps({"questions": results}, indent=2))

def bioasq(test_data='phaseB_3b_01.json',
           output_file='bioasq-out-simple.json'):
    """Produce results ready for submission to BioASQ"""
    print("Loading training and test data for BioASQ run")
    testset = load_test_data(test_data)

    print("Processing test data")
    answers = yield_bioasq_answers(testset,
                                   nanswers={"summary": 6,
                                             "factoid": 2,
                                             "yesno": 2,
                                             "list": 3})
    result = {"questions":[a for a in answers]}
    print("Saving results in file %s" % output_file)
    with open(output_file,'w') as f:
        f.write(json.dumps(result, indent=2))

def yield_bioasq_answers(test, nanswers=3):
    """Yield answer of each record for BioASQ shared task"""
    questions = [r['body'] for r in test]
    q_embedding = np.array([reduce_embeddings(word2vec.vectors(tokenize(q,stem=False)),
                                              dimension=word2vec.DIMENSION)
                   for q in questions])
    q_embedding = my_normalize(q_embedding)

    text = [t for r in test for t in yield_candidate_snippets(r)]
    #print(text)

    t_embedding = np.array([reduce_embeddings(word2vec.vectors(tokenize(t,stem=False)),
                                              dimension=word2vec.DIMENSION)
                            for t in text])
    t_embedding = my_normalize(t_embedding)

    iq = 0
    it = 0
    itemnumber = 0
    for r in test:
        test_id = r['id']
        itemnumber += 1
        startit = it
        # snippets = r['snippets']
        snippets = [t for t in yield_candidate_snippets(r)]
        it += len(snippets)

        if type(nanswers) == dict:
            n = nanswers[r['type']]
        else:
            n = nanswers

        w_cos_embeddings = np.array([cosine_similarity([x],[q_embedding[iq,:]])
                                      for x in t_embedding[startit:it]])

        iq += 1
        if len(snippets) > 0:
            w_cos_embeddings = w_cos_embeddings[:,0,0]
        else:
            # pass
            print("Warning: No snippets found in this record:")
            print(r)

        if r['type'] == "yesno":
            exactanswer = "yes"
        else:
            exactanswer = ""

        yield {"id": test_id,
               "ideal_answer": " ".join([
                   snippets[x] for x in topn(w_cos_embeddings,n)
               ]),
               "exact_answer": exactanswer}

if __name__ == "__main__":
    import doctest
    doctest.testmod()

#    import sys
#    sys.exit()

#    bioasq()

#    import sys
#    sys.exit()

    rouge_names = ['firstn.xml','random.xml','rouge_cos_q_lsa_t_lsa.xml','rouge_cos_embeddings.xml']

    import codecs
    import os
    #import random
    from sklearn.model_selection import KFold
    from subprocess import Popen, PIPE

    rouge = True

    print("Loading training data")
    #dataset = loaddata('BioASQ-trainingDataset5b.json',qtype='summary')
    #dataset = loaddata('BioASQ-trainingDataset5b.json',qtype='factoid')
    #dataset = loaddata('BioASQ-trainingDataset5b.json',qtype='yesno')
    #dataset = loaddata('BioASQ-trainingDataset5b.json',qtype='list')
    dataset = loaddata('BioASQ-training8b.json')

    indices = list(range(len(dataset)))
    random.seed(1234)
    random.shuffle(indices)

    rouge_results = []
    fold = 0
    kf = KFold(n_splits=10)
    for (traini,testi) in kf.split(indices):
        fold += 1

        #if fold > 2:
        #    break

        print("Cross-validation Fold %i" % fold)
        train = [dataset[indices[i]] for i in traini]
        test = [dataset[indices[i]] for i in testi]

        print("Fitting tfidf")
        tfidf = TfidfVectorizer(input='content',tokenizer=tokenize)
        tfidf_features = tfidf.fit_transform(yield_text(train))
    #    tfidf_features = tfidf.fit_transform(yield_text(test))

        lsa_components = 200
        print("Fitting LSA with %i components" % lsa_components)
        lsa = TruncatedSVD(n_components=lsa_components)
        lsa.fit(tfidf_features)

        print("Finding key sentences")
        answers = yield_answers(test,tfidf,lsa,
                                nanswers={"summary": 6,
                                          "factoid": 2,
                                          "yesno": 2,
                                          "list": 3})
        if rouge:
            frouge = []
            for fname in rouge_names:
                f = open(fname,'w')
                f.write('<ROUGE-EVAL version="1.0">\n')
                frouge.append(f)

        allanswers = []
        for i in range(len(test)):
            theanswers = next(answers)
            allanswers.append(theanswers)

            if rouge:
                for fi in range(len(frouge)):
                    frouge[fi].write("""<EVAL ID="%i">
     <MODEL-ROOT>
     ../rouge/models
     </MODEL-ROOT>
     <PEER-ROOT>
     ../rouge/summaries
     </PEER-ROOT>
     <INPUT-FORMAT TYPE="SEE">
     </INPUT-FORMAT>
    """ % (i))
                    frouge[fi].write(""" <PEERS>
      <P ID="A">summary%i_%i.txt</P>
     </PEERS>
     <MODELS>
    """ % (fi,i))
                    with codecs.open(os.path.join('..','rouge','summaries',
                                                  'summary%i_%i.txt' % (fi,i)),
                                 'w', 'utf-8') as fout:
                        a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</a></body>\n</html>'.format(" ".join(theanswers[fi]))
                        fout.write(a+'\n')
                        # fout.write('\n'.join([s for t in theanswers[fi]
                        #                       for s in nltk.sent_tokenize(t)])+'\n')

                if type(test[i]['ideal_answer']) == list:
                    ideal_answers = test[i]['ideal_answer']
                else:
                    ideal_answers = [test[i]['ideal_answer']]
                for j in range(len(ideal_answers)):
                    for fi in range(len(frouge)):
                        frouge[fi].write('  <M ID="%i">ideal_answer%i_%i.txt</M>\n' % (j,i,j))
                    with codecs.open(os.path.join('..','rouge','models',
                                                  'ideal_answer%i_%i.txt' % (i,j)),
                                     'w', 'utf-8') as fout:
                        a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</a></body>\n</html>'.format(ideal_answers[j])
                        # a='\n'.join(nltk.sent_tokenize(ideal_answers[j]))
                        fout.write(a+'\n')
                for f in frouge:
                    f.write(""" </MODELS>
    </EVAL>
    """)

        if rouge:
            for f in frouge:
                f.write('</ROUGE-EVAL>\n')
                f.close()
            rouge_results.append(dict())
            for fname in rouge_names:
                print("Calling ROUGE", fname)
                ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ../rouge/RELEASE-1.5.5/data -c 95 -2 4 -u -x -n 4 -a ' + fname
                # ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ../rouge/RELEASE-1.5.5/data -a -n 2 -2 4 -U '+fname
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
                     'SU4':float(lines[19].split()[3])}
                rouge_results[-1][fname] = F

        print("Average number of sentences:")
        print('All', np.average([len(x['snippets']) for x in test
                                                    if 'snippets' in x]))
        for i in range(len(rouge_names)):
            print(rouge_names[i], np.average([len(x[i]) for x in allanswers]))


    for experiment in rouge_names:
        print("Experiment", experiment)
        print("%5s %7s %7s" % ('', 'N-2', 'SU4'))
        for i in range(len(rouge_results)):
            print("%5i %1.5f %1.5f" % (i+1,rouge_results[i][experiment]['N-2'],rouge_results[i][experiment]['SU4']))
        mean_N2 = np.average([rouge_results[i][experiment]['N-2']
                              for i in range(len(rouge_results))])
        mean_SU4 = np.average([rouge_results[i][experiment]['SU4']
                               for i in range(len(rouge_results))])
        print("%5s %1.5f %1.5f" % ("mean",mean_N2,mean_SU4))
        stdev_N2 = np.std([rouge_results[i][experiment]['N-2']
                              for i in range(len(rouge_results))])
        stdev_SU4 = np.std([rouge_results[i][experiment]['SU4']
                               for i in range(len(rouge_results))])
        print("%5s %1.5f %1.5f" % ("stdev",stdev_N2,stdev_SU4))
        print()
