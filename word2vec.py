"""Interface to BioASQ's word2vec vectors
Author: Diego Molla
Created: 27 August 2015
"""

import numpy as np
#import pickle
import os.path
import sqlite3

DIMENSION = 200

VECTORS = 'allMeSH_2016_%i.vectors.txt' % DIMENSION
DB = 'word2vec_cnn_%i.db' % DIMENSION

if not os.path.exists(DB):
    print("Creating database of vectors %s" % DB)
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("CREATE TABLE vectors (word unicode, data unicode)")
    with open(VECTORS) as v:
        nwords = int(v.readline().strip().split()[0])
        print("Processing %i words" % nwords)
        for i in range(nwords):
            vector = v.readline()
            windex = vector.index(" ")
            w = vector[:windex].strip()
            d = vector[windex:].strip()
            assert len(d.split()) == DIMENSION
            #if i < 5:
            #    print(w)
            #    print(d)
            c.execute("INSERT INTO vectors VALUES (?,?)",(w,d))
    c.execute("CREATE INDEX idx ON vectors (word)")
    conn.commit()
    conn.close()

vectordb = sqlite3.connect(DB)

# print "Reading word2vec word list"
# wordlist = dict()
# with open(INDEX) as f:
#     i = 0
#     for l in f.readlines():
#         ll = l.strip()
#         if not wordlist.has_key(ll):
#             wordlist[ll] = i
#         i += 1

# print "Reading word2vec vectors"
# with open(VECTORS) as f:
#     vectorlist = []
#     for l in f.readlines():
#         vectorlist.append([float(x) for x in l.strip().split()])
# vectorlist = [[0]*len(vectorlist[0])] + vectorlist

class Vectors:
    "A class that extracts the word vectors on the run"
    def __init__(self,text, tokenizer=None):
        "If tokenizer is not defined, we assume that the text is tokenised already"
        self.text = text
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self,index):
        if isinstance(index, slice):
            return self._getslice(index)
        if isinstance(index, list) or isinstance(index, np.ndarray):
            result = []
            for i in index:
                s = self.text[i]
                if self.tokenizer:
                    tokens = self.tokenizer(s)
                else:
                    tokens = s
                result.append(vectors(tokens))
            return result
        else:
            s = self.text[index]
            if self.tokenizer:
                tokens = self.tokenizer(s)
            else:
                tokens = s
            return vectors(tokens)

    def _getslice(self,indices):
        result = []
        for s in self.text[indices]:
            if self.tokenizer:
                tokens = self.tokenizer(s)
            else:
                tokens = s
            result.append(vectors(tokens))
        return result


def vectors(words,returndict=False):
    """Return the vectors of the list of words.
Words without vector are ignored.
"""
    c = vectordb.cursor()
    if returndict:
        result = dict()
    else:
        result = []

    # return result # empty vectors
        
    for w in words:
        c.execute("SELECT data FROM vectors INDEXED BY idx WHERE word=?",(w,))
        r = c.fetchall()
        if len(r) == 0:
            continue
#            result.append(None)
        embedding = [float(x) for x in r[0][0].split()]
        if returndict:
            result[w] = np.array(embedding)
        else:
            result.append(embedding)

    return result

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    v = vectors(['disease','###','tttrreewws','###','book'])
    print("Vectors of",['disease','###','tttrreewws','###','book'])
    for x in v:
        if x:
            print('[%f, %f, %f, ...]' % tuple(x[:3]))
        else:
            print("Unknown word")
