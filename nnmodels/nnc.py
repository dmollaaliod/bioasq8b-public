"NN models for classification"
import re
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sqlite3
import random
import os
import glob
import sys
from multiprocessing import Pool
from functools import partial

from transformers import BertTokenizer, TFBertModel
from sklearn.metrics import mean_squared_error

from nnmodels import compare


# Bug fix; see https://github.com/tensorflow/tensorflow/issues/24496
# This code fixes the error message "UnknownError:  [_Derived_]  Fail to find the dnn implementation."
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = InteractiveSession(config=config)
# End bug fix

EMBEDDINGS = 100
VECTORS = 'allMeSH_2016_%i.vectors.txt' % EMBEDDINGS
DB = 'word2vec_%i.db' % EMBEDDINGS
MAX_NUM_SNIPPETS = 50
SENTENCE_LENGTH = 300

print("LSTM using sentence length=%i" % SENTENCE_LENGTH)
print("LSTM using embedding dimension=%i" % EMBEDDINGS)

with open(VECTORS) as v:
    VOCABULARY = int(v.readline().strip().split()[0]) + 2

if not os.path.exists(DB):
    print("Creating database of vectors %s" % DB)
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE vectors (word unicode,
                                       word_index integer,
                                       data unicode)""")
    with open(VECTORS, encoding='utf-8') as v:
        nwords = int(v.readline().strip().split()[0])
        print("Processing %i words" % nwords)
        zeroes = " ".join("0"*EMBEDDINGS)
        # Insert PAD and UNK special words with zeroes
        c.execute("INSERT INTO vectors VALUES (?, ?, ?)", ('PAD', 0, zeroes))
        c.execute("INSERT INTO vectors VALUES (?, ?, ?)", ('UNK', 1, zeroes))
        for i in range(nwords):
            vector = v.readline()
            windex = vector.index(" ")
            w = vector[:windex].strip()
            d = vector[windex:].strip()
            assert len(d.split()) == EMBEDDINGS
            #if i < 5:
            #    print(w)
            #    print(d)
            c.execute("INSERT INTO vectors VALUES (?, ?, ?)", (w, i+2, d))
    c.execute("CREATE INDEX word_idx ON vectors (word)")
    conn.commit()
    conn.close()

#vectordb = sqlite3.connect(DB)

def sentences_to_ids(sentences, sentence_length=SENTENCE_LENGTH):
    """Convert each sentence to a list of word IDs.

Crop or pad to 0 the sentences to ensure equal length if necessary.
Words without ID are assigned ID 1.
>>> sentences_to_ids([['my','first','sentence'],['my','ssecond','sentence'],['yes']], 2)
(([11095, 121], [11095, 1], [21402, 0]), (2, 2, 1))
"""
    return tuple(zip(*map(partial(one_sentence_to_ids,
                                  sentence_length=sentence_length),
                          sentences)))

def one_sentence_to_ids(sentence, sentence_length=SENTENCE_LENGTH):
    """Convert one sentence to a list of word IDs."

Crop or pad to 0 the sentences to ensure equal length if necessary.
Words without ID are assigned ID 1.
>>> one_sentence_to_ids(['my','first','sentence'], 2)
([11095, 121], 2)
>>> one_sentence_to_ids(['my','ssecond','sentence'], 2)
([11095, 1], 2)
>>> one_sentence_to_ids(['yes'], 2)
([21402, 0], 1)
"""
    vectordb = sqlite3.connect(DB)
    c = vectordb.cursor()
    word_ids = []
    for w in sentence:
        if len(word_ids) >= sentence_length:
            break
        c.execute("""SELECT word_index, word
                  FROM vectors
                  INDEXED BY word_idx
                  WHERE word=?""", (w, ))
        r = c.fetchall()
        if len(r) > 0:
            word_ids.append(r[0][0])
        else:
            word_ids.append(1)
    # Pad with zeros if necessary
    num_words = len(word_ids)
    if num_words < sentence_length:
        word_ids += [0]*(sentence_length-num_words)
    vectordb.close()
    return word_ids, num_words

def parallel_sentences_to_ids(sentences, sentence_length=SENTENCE_LENGTH):
    """Convert each sentence to a list of word IDs.

Crop or pad to 0 the sentences to ensure equal length if necessary.
Words without ID are assigned ID 1.
>>> parallel_sentences_to_ids([['my','first','sentence'],['my','ssecond','sentence'],['yes']], 2)
(([11095, 121], [11095, 1], [21402, 0]), (2, 2, 1))
"""
    with Pool() as pool:
        return tuple(zip(*pool.map(partial(one_sentence_to_ids,
                                           sentence_length=sentence_length),
                                   sentences)))

def snippets_to_ids(snippets, sentence_length, max_num_snippets=MAX_NUM_SNIPPETS):
    """Convert the snippets to lists of word IDs.
    >>> snippets_to_ids([['sentence', 'one'], ['sentence'], ['two']], 3, 2)
    (([12205, 68, 0], [12205, 0, 0]), (2, 1))
    >>> snippets_to_ids([['sentence', 'three']], 3, 2)
    (([12205, 98, 0], [0, 0, 0]), (2, 0))
    """
    # Pad to the maximum number of snippets
    working_sample = snippets[:max_num_snippets]
    #print("Number of snips: %i" % len(sample))
    if len(working_sample) < max_num_snippets:
        working_sample += [[]] * (max_num_snippets-len(working_sample))

    # Convert to word IDs
    return sentences_to_ids(working_sample, sentence_length)

def parallel_snippets_to_ids(batch_snippets,
                             sentence_length,
                             max_num_snippets=MAX_NUM_SNIPPETS):
    """Convert the batch of snippets to lists of word IDs.
    >>> parallel_snippets_to_ids([[['sentence', 'one'], ['sentence'], ['two']],[['sentence', 'three']]], 3, 2)
    ((([12205, 68, 0], [12205, 0, 0]), ([12205, 98, 0], [0, 0, 0])), ((2, 1), (2, 0)))
    """
    with Pool() as pool:
        return tuple(zip(*pool.map(partial(snippets_to_ids,
                                           sentence_length=sentence_length,
                                           max_num_snippets=max_num_snippets),
                                   batch_snippets)))

def parallel_sentences_to_bert_ids(batch_snippets, 
                                   sentence_length,
                                   tokenizer,
                                   max_num_snippets=MAX_NUM_SNIPPETS):
    "Convert the text to indices and pad-truncate to the maximum number of words"
    with Pool() as pool:
        return pool.map(partial(tokenizer.encode,
                                add_special_tokens=True, max_length=sentence_length, pad_to_max_length=True),
                         batch_snippets)

def embeddings_one_sentence(row):
    return [float(x) for x in row[1].split()]

class BasicNN:
    """A simple NN classifier"""
    def __init__(self, sentence_length=SENTENCE_LENGTH, batch_size=128, embeddings=True,
                 hidden_layer=0, build_model=False):
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.embeddings = None
        self.hidden_layer = hidden_layer
        self.build_model = build_model
        self.cleantext = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                     t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                                   '').strip().lower()).split()
        if embeddings:
            vectordb = sqlite3.connect(DB)
            print("Database %s opened" % DB)
            c = vectordb.cursor()
            c_iterator = c.execute("""SELECT word_index, data
                                      FROM vectors""")
            print("Loading word embeddings")
            with Pool() as pool:
                self.embeddings = pool.map(embeddings_one_sentence,
                                           c_iterator)
            print("Word embeddings loaded")
            vectordb.close()

    def name(self):
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "BasicNN%s%s" % (str_embeddings, str_hidden)


    def __build_model__(self, learningrate=0.001, keep_prob=0.5, verbose=1):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(VOCABULARY, EMBEDDINGS))
        model.add(keras.layers.GlobalAveragePooling1D())
        if self.hidden_layer > 0:
            model.add(keras.layers.Dense(self.hidden_layer, activation=tf.nn.relu))
        model.add(keras.layers.Dropout(1 - keep_prob)) # Is this correct??
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

        if self.embeddings != None:
            model.layers[0].set_weights([np.array(self.embeddings)])
            model.layers[0].trainable = False

        model.compile(optimizer=tf.optimizers.Adam(learningrate), #train.AdamOptimizer(learningrate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        if verbose:
            model.summary()

        return model

    def restore(self, savepath):
        model = self.__build_model__()
        model.load_weights(savepath)
        self.model=model
        print("Model restored from file: %s" % savepath)

    def fit(self, X_train, Q_train, Y_train,
            X_positions = [],
            validation_data = None,
            verbose=2, nb_epoch=3,
            learningrate=0.001, dropoutrate=0.5, savepath=None,
            restore_model=False):
        """ Q_train and X_positions are not used in this function, 
        they are here for API compatibility."""

        if restore_model:
            print("Restoring BasicNN model from %s" % savepath)
            self.model.load_weights(savepath)
            return self.test(X_train, Q_train, Y_train, X_positions=X_positions)

        assert(len(X_train) == len(Y_train))

        # Training loop
        print("Extracting sentence IDs")
        X, sequence_lengths = parallel_sentences_to_ids(X_train, self.sentence_length)
        print("Sentence IDs extracted")

        if validation_data:
            X_val, Q_val, Y_val, Xpos_val = validation_data
            X_val, sequence_lengths_x_val = parallel_sentences_to_ids(X_val, self.sentence_length)
            val_data = (np.array(X_val), np.array(Y_val))
        else:
            val_data = None

        return self.__fit__(X, Y_train,
                            val_data,
                            verbose, nb_epoch, learningrate,
                            dropoutrate, savepath)

    def __fit__(self, X, Y,
                val_data,
                verbose, nb_epoch, learningrate, keep_prob,
                savepath=None):
        self.model = self.__build_model__(learningrate, keep_prob, verbose)

        X = np.array(X)
        #sequence_lengths = np.array(sequence_lengths)

        if savepath:
            callbacks = [tf.keras.callbacks.ModelCheckpoint(savepath,
                                                            save_weights_only=True,
                                                            verbose=1)]
        else:
            callbacks = []

        history = self.model.fit(X, Y, 
                                 validation_data=val_data,
                                 verbose=verbose,
                                 epochs=nb_epoch, callbacks=callbacks, batch_size=self.batch_size)

        return history.history

    def predict(self, X_topredict, Q_topredict, X_positions=[]):
        X, sequence_lengths = parallel_sentences_to_ids(X_topredict, self.sentence_length)
        return self.model.predict(X)

    def test(self, X_test, Q_test, Y_test, X_positions=[]):
        X, sequence_lengths = parallel_sentences_to_ids(X_test, self.sentence_length)
        return self.__test__(X, Y_test)

    def __test__(self, X, Y):
        X = np.array(X)
        #return self.model.fit(X, Y, epochs=3, batch_size=self.batch_size)
        return self.model.evaluate(X, Y, batch_size=self.batch_size)

class BasicBERT(BasicNN):
    """A simple BERT classifier"""
    def __init__(self, batch_size=32, hidden_layer=0, build_model=False):
        self.batch_size = batch_size
        self.sentence_length = 250 # as observed in exploratory file bert-exploration.ipynb
        self.hidden_layer = hidden_layer
        self.build_model = build_model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cleantext = lambda t: t

    def name(self):
        return "BERT"

    def __build_model__(self, learningrate=None, keep_prob=0.5, verbose=1):
        inputs = keras.layers.Input(shape=(self.sentence_length,), dtype=tf.int32)
        bert = TFBertModel.from_pretrained('bert-base-uncased', trainable=False)(inputs)[0]
        average_pooling = keras.layers.GlobalAveragePooling1D()(bert)
        if self.hidden_layer > 0:
            hidden = keras.layers.Dense(self.hidden_layer, activation=tf.nn.relu)(average_pooling)
        else:
            hidden = average_pooling
        dropout = keras.layers.Dropout(1 - keep_prob)(hidden)
        outputs = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dropout)
        model = keras.models.Model(inputs, outputs)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        if verbose:
            model.summary()
        
        return model

    def fit(self, X_train, Q_train, Y_train,
            X_positions = [],
            validation_data=None,
            verbose=2, nb_epoch=3,
            dropoutrate=0.5,
            savepath=None, restore_model=False):

        if restore_model:
            print("Restoring BERT model from %s" % savepath)
            self.restore(savepath)
            return self.test(X_train, Q_train, Y_train, X_positions)

        # Training loop
        print("Extracting BERT sentence IDs")
        X = parallel_sentences_to_bert_ids(X_train, self.sentence_length, self.tokenizer)
        print("Sentence BERT IDs extracted")

        if validation_data:
            X_val, Q_val, Y_val, Xpos_val = validation_data
            X_val = parallel_sentences_to_bert_ids(X_val, self.sentence_length, self.tokenizer)
            val_data = (np.array(X_val), np.array(Y_val))
        else:
            val_data = None

        return self.__fit__(X, Y_train,
                            val_data,
                            verbose, nb_epoch, None, dropoutrate, savepath)

    def predict(self, X_topredict, Q_topredict, X_positions=[]):
        X = parallel_sentences_to_bert_ids(X_topredict, self.sentence_length, self.tokenizer)
        return self.model.predict(X)

    def test(self, X_test, Q_test, Y_test, X_positions=[]):
        X = parallel_sentences_to_bert_ids(X_test, self.sentence_length, self.tokenizer)
        return self.__test__(X, Y_test)


class Similarities(BasicNN):
    """A classifier that incorporates similarity operations"""
    def __init__(self, sentence_length=SENTENCE_LENGTH, batch_size=128, embeddings=True,
                 hidden_layer=0, build_model=False, comparison=compare.SimMul(), positions=False, 
                 regression=False, siamese=False):
        BasicNN.__init__(self, sentence_length, batch_size, embeddings,
                 hidden_layer, build_model)
        self.comparison = comparison
        self.positions = positions
        self.regression_model = regression
        self.siamese = siamese

    def name(self):
        if self.siamese:
            str_siamese = "Siamese"
        else:
            str_siamese = ""
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "Mean%s%s%s%s%s" % (str_siamese, self.comparison.name, str_embeddings, str_positions, str_hidden)

    def embedding_reduction(self, keep_prob):
        "Return the sentence embeddings based on word embeddings"
        input_layer = keras.layers.Input(shape=(self.sentence_length, EMBEDDINGS))
        embedding_layer = keras.layers.GlobalAveragePooling1D()(input_layer)
        return keras.models.Model(input_layer, embedding_layer)

    def __build_model__(self,
            #embeddings_lambda=10,
                        learningrate=0.001, keep_prob=0.5, verbose=1):

        # Sentence
        X_input = keras.layers.Input(shape=(self.sentence_length,), name='X')
        if self.embeddings == None:
            embedding_s = keras.layers.Embedding(VOCABULARY, EMBEDDINGS)(X_input)
        else:
            embedding_s = keras.layers.Embedding(VOCABULARY, EMBEDDINGS,
                                                 weights=[np.array(self.embeddings)],
                                                 trainable=False)(X_input)
        embedding_reduction_layer = self.embedding_reduction(keep_prob)
        embedding_s_reduction = embedding_reduction_layer(embedding_s)
        X_dropout = keras.layers.Dropout(1 - keep_prob)(embedding_s_reduction)

        # Question
        Q_input = keras.layers.Input(shape=(self.sentence_length,), name='Q')
        if self.embeddings == None:
            embedding_q = keras.layers.Embedding(VOCABULARY, EMBEDDINGS)(Q_input)
        else:
            embedding_q = keras.layers.Embedding(VOCABULARY, EMBEDDINGS,
                                                 weights=[np.array(self.embeddings)],
                                                 trainable=False)(Q_input)
        if self.siamese:
            embedding_q_reduction = embedding_reduction_layer(embedding_q)
        else:
            embedding_q_reduction_layer = self.embedding_reduction(keep_prob)
            embedding_q_reduction = embedding_q_reduction_layer(embedding_q)

        Q_output = keras.layers.Dropout(1 - keep_prob)(embedding_q_reduction)

        # Similarity
        sim = keras.layers.Multiply()([X_dropout, Q_output])

        # Sentence position
        if self.positions:
            positions = [keras.layers.Input(shape=(1,), name='X2')]
        else:
            positions = []

        # Concatenate all inputs
        all_inputs = keras.layers.Concatenate()([X_dropout, sim] + positions)

        # Hidden layer
        if self.hidden_layer > 0:
            hidden = keras.layers.Dense(self.hidden_layer, activation=tf.nn.relu)(all_inputs)
        else:
            hidden = all_inputs
        dropout_2 = keras.layers.Dropout(1 - keep_prob)(hidden)

        return self.top_layer(dropout_2, X_input, Q_input, positions, learningrate, verbose)

    def top_layer(self, dropout_2, X_input, Q_input, positions, learningrate, verbose):
        # Final prediction
        if not(self.regression_model):
            print("top_layer: classification model (sigmoid activation)")
            output = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dropout_2)
            model = keras.models.Model([X_input, Q_input] + positions, output)
            model.compile(optimizer=tf.optimizers.Adam(learningrate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        else:
            print("top_layer: regression model (linear activation)")
            output = keras.layers.Dense(1, activation=None)(dropout_2)
            model = keras.models.Model([X_input, Q_input] + positions, output)
            model.compile(optimizer=tf.optimizers.Adam(learningrate),
                                  loss='mean_squared_error',
                                  metrics=['accuracy'])

        if verbose:
            model.summary()

        return model

    def fit(self, X_train, Q_train, Y_train,
            X_positions = [],
            validation_data = None,
            verbose=2, nb_epoch=3,
            learningrate=0.001, dropoutrate=0.5, savepath=None,
            restore_model=False):

        if restore_model:
            print("Restoring Similarities model from %s" % savepath)
            #self.model.load_weights(savepath)
            self.restore(savepath)
            return self.test(X_train, Q_train, Y_train, X_positions=X_positions)

        assert(len(X_train) == len(Q_train))
        assert(len(X_train) == len(Y_train))

        # Training loop
        X, sequence_lengths_x = parallel_sentences_to_ids(X_train, self.sentence_length)
        Q, sequence_lengths_q = parallel_sentences_to_ids(Q_train, self.sentence_length)

        if validation_data:
            X_val, Q_val, Y_val, Xpos_val = validation_data
            X_val, sequence_lengths_x_val = parallel_sentences_to_ids(X_val, self.sentence_length)
            Q_val, sequence_lengths_q_val = parallel_sentences_to_ids(Q_val, self.sentence_length)
            val_data = ({'X': np.array(X_val),
                         'Q': np.array(Q_val),
                         'X2': np.array(Xpos_val)},
                        np.array(Y_val))
        else:
            val_data = None

        return self.__fit__(X, X_positions, Q, Y_train,
                            val_data,
                            verbose, nb_epoch, learningrate,
                            dropoutrate, savepath)

    def __fit__(self, X, X_positions, Q, Y,
                val_data,
                verbose, nb_epoch, learningrate, keep_prob,
                savepath=None):
        self.model = self.__build_model__(learningrate, keep_prob, verbose)

        X = np.array(X)
        Q = np.array(Q)

        #savepath = None

        if verbose == 1 or verbose == 2:
            verbose = 3 -  verbose
        history = self.model.fit({'X':X, 'Q':Q, 'X2':X_positions}, Y,
                                 validation_data=val_data,
                                 verbose=verbose,
                                 epochs=nb_epoch, callbacks=[], batch_size=self.batch_size)

        if savepath:
            print ("Save: %s" % savepath)
            self.model.save_weights(savepath)
        return history.history

    def predict(self, X_topredict, Q_topredict, X_positions=[]):
        assert(len(X_topredict) == len(Q_topredict))
        if len(X_topredict) == 0:
            print("WARNING: Data to predict is an empty list")
            return []
        X, sequence_lengths_x = parallel_sentences_to_ids(X_topredict, self.sentence_length)
        Q, sequence_lengths_q = parallel_sentences_to_ids(Q_topredict, self.sentence_length)
        X = np.array(X)
        Q = np.array(Q)
        return self.model.predict({'X':X, 'Q': Q, 'X2': np.array(X_positions, dtype=np.float32)})

    def test(self, X_test, Q_test, Y, X_positions=[]):
        assert(len(X_test) == len(Q_test))
        assert(len(X_test) == len(Y))

        X, sequence_lengths_x = parallel_sentences_to_ids(X_test, self.sentence_length)
        Q, sequence_lengths_q = parallel_sentences_to_ids(Q_test, self.sentence_length)
        return self.__test__(X,
                             Q,
                             Y,
                             X_positions)

    def __test__(self, X,
                       Q,
                       Y,
                       X_positions):
        X = np.array(X)
        Q = np.array(Q)
        return self.model.evaluate({'X':X, 'Q': Q, 'X2': X_positions}, Y, batch_size=self.batch_size)

class SimilaritiesBERT(Similarities):
    """A classifier that uses BERT for similarities"""
    def __init__(self, batch_size=32, hidden_layer=0, build_model=False,
                 comparison=compare.SimMul(), positions=False, regression=False, trainable=False,
                 siamese=False):
        self.batch_size = batch_size
        self.trainable = trainable
        self.sentence_length = 250
        self.hidden_layer = hidden_layer
        self.build_model = build_model
#        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = self.load_tokenizer()
        self.cleantext = lambda t: t
        self.comparison = comparison
        self.positions = positions
        self.regression_model = regression
        self.siamese = siamese

    def load_tokenizer(self):
#        return BertTokenizer.from_pretrained('bert-tokenizer-base-uncased')
        return BertTokenizer.from_pretrained('bert-base-uncased')

    def load_bert_model(self):
#        return TFBertModel.from_pretrained('bert-tfmodel-base-uncased')
        return TFBertModel.from_pretrained('bert-base-uncased')

    def name(self):
        if self.siamese:
            str_siamese = "Siamese"
        else:
            str_siamese = ""
        if self.trainable:
            str_trainable = "Trainable"
        else:
            str_trainable = ""
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "BERT%s%sMean%s%s%s" % (str_siamese, str_trainable, self.comparison.name, str_positions, str_hidden)

    def restore(self, savepath):
        model, _X_bert_layer, _Q_bert_layer = self.__build_model__()
        model.load_weights(savepath)
        self.model=model
        print("Model restored from file: %s" % savepath)

    def embedding_reduction(self, keep_prob):
        "Return the sentence embeddings based on word embeddings"
        input_layer = keras.layers.Input(shape=(self.sentence_length, 768))
        embedding_layer = keras.layers.GlobalAveragePooling1D()(input_layer)
        return keras.models.Model(input_layer, embedding_layer)

    
    def __build_model__(self,
                        learningrate=None,
                        keep_prob=0.5,
                        verbose=1):
        # Sentence
        X_input = keras.layers.Input(shape=(self.sentence_length,), dtype=tf.int32, name='X')
#        X_bert = TFBertModel.from_pretrained('bert-base-uncased', trainable=self.trainable)(X_input)[0]
        X_bert_layer = self.load_bert_model()
        X_bert = X_bert_layer(X_input)[0]

        embedding_reduction_layer = self.embedding_reduction(keep_prob)
        X_embedding_reduction = embedding_reduction_layer(X_bert)
        X_dropout = keras.layers.Dropout(1 - keep_prob)(X_embedding_reduction)

        # Question
        Q_input = keras.layers.Input(shape=(self.sentence_length,), dtype=tf.int32, name='Q')
#        Q_bert = TFBertModel.from_pretrained('bert-base-uncased', trainable=self.trainable)(Q_input)[0]
#        Q_bert = TFBertModel.from_pretrained('bert-tfmodel-base-uncased', trainable=self.trainable)(Q_input)[0]
        Q_bert_layer = self.load_bert_model()
        Q_bert = Q_bert_layer(Q_input)[0]

        if self.siamese:
            Q_embedding_reduction = embedding_reduction_layer(Q_bert)
        else:    
            embedding_q_reduction_layer = self.embedding_reduction(keep_prob)
            Q_embedding_reduction = embedding_q_reduction_layer(Q_bert)

        Q_dropout = keras.layers.Dropout(1 - keep_prob)(Q_embedding_reduction)

        # Similarity
        sim = keras.layers.Multiply()([X_dropout, Q_dropout])

        # Sentence position
        if self.positions:
            positions = [keras.layers.Input(shape=(1,), name='X2')]
        else:
            positions = []

        # Concatenate all inputs
        all_inputs = keras.layers.Concatenate()([X_dropout, sim] + positions)

        # Hidden layer
        if self.hidden_layer > 0:
            hidden = keras.layers.Dense(self.hidden_layer, activation=tf.nn.relu)(all_inputs)
        else:
            hidden = all_inputs
        dropout_2 = keras.layers.Dropout(1 - keep_prob)(hidden)
        
        # Top layer
        if not(self.regression_model):
            print("top_layer: classification model (sigmoid activation)")
            output = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dropout_2)
            model = keras.models.Model([X_input, Q_input] + positions, output)
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        else:
            print("top_layer: regression model (linear activation)")
            output = keras.layers.Dense(1, activation=None)(dropout_2)
            model = keras.models.Model([X_input, Q_input] + positions, output)
            model.compile(optimizer='adam',
                          loss='mean_squared_error',
                          metrics=['accuracy'])

        if verbose:
            model.summary()

        return model, X_bert_layer, Q_bert_layer


    def __fit__(self, X, X_positions, Q, Y,
                val_data,
                verbose, nb_epoch, learningrate, keep_prob,
                savepath=None):
        self.model, X_bert_layer, Q_bert_layer = self.__build_model__(learningrate, keep_prob, verbose)
        if self.trainable:
            # Warmup stage that fine-tunes BERT weights
            print("Warmup stage that fine-tunes BERT weights, with batch size 8")
            X_bert_layer.trainable = True
            Q_bert_layer.trainable = True

            if self.regression_model:
                self.model.compile(optimizer='adam',
                                   loss='mean_squared_error',
                                   metrics=['accuracy'])
            else:
                self.model.compile(optimizer='adam',
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])

            if verbose:
                self.model.summary()
                
            X = np.array(X)
            Q = np.array(Q)

            #savepath = None

            if verbose == 1 or verbose == 2:
                verbose_fit = 3 -  verbose
            history = self.model.fit({'X':X, 'Q':Q, 'X2':X_positions}, Y,
                                     validation_data=val_data,
                                     verbose=verbose_fit,
                                     epochs=1, callbacks=[], batch_size=8)

            nb_epoch -= 1

        print("Training stage, with batch size", self.batch_size)
        X_bert_layer.trainable = False
        Q_bert_layer.trainable = False
        if self.regression_model:
            self.model.compile(optimizer='adam',
                               loss='mean_squared_error',
                               metrics=['accuracy'])
        else:
            self.model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

        if verbose:
            self.model.summary()

        X = np.array(X)
        Q = np.array(Q)

        #savepath = None

        if verbose == 1 or verbose == 2:
            verbose_fit = 3 -  verbose
        history = self.model.fit({'X':X, 'Q':Q, 'X2':X_positions}, Y,
                                 validation_data=val_data,
                                 verbose=verbose_fit,
                                 epochs=nb_epoch, callbacks=[], batch_size=self.batch_size)

            
        if savepath:
            print ("Save: %s" % savepath)
            self.model.save_weights(savepath)
            
        return history.history

    def fit(self, X_train, Q_train, Y_train,
            X_positions = [],
            validation_data = None,
            verbose=2, nb_epoch=3,
            dropoutrate=0.5, savepath=None,
            restore_model=False):

        if restore_model:
            print("Restoring  BERT Similarities model from %s" % savepath)
            #self.model.load_weights(savepath)
            self.restore(savepath)
            return self.test(X_train, Q_train, Y_train, X_positions=X_positions)

        assert(len(X_train) == len(Q_train))
        assert(len(X_train) == len(Y_train))

        # Training loop
        X = parallel_sentences_to_bert_ids(X_train, self.sentence_length, self.tokenizer)
        Q = parallel_sentences_to_bert_ids(Q_train, self.sentence_length, self.tokenizer)

        if validation_data:
            X_val, Q_val, Y_val, Xpos_val = validation_data
            X_val = parallel_sentences_to_bert_ids(X_val, self.sentence_length, self.tokenizer)
            Q_val = parallel_sentences_to_bert_ids(Q_val, self.sentence_length, self.tokenizer)
            val_data = ({'X': np.array(X_val),
                         'Q': np.array(Q_val),
                         'X2': np.array(Xpos_val)},
                        np.array(Y_val))
        else:
            val_data = None

        return self.__fit__(X, X_positions, Q, Y_train,
                            val_data,
                            verbose, nb_epoch, None,
                            dropoutrate, savepath)

    def predict(self, X_topredict, Q_topredict, X_positions=[]):
        assert(len(X_topredict) == len(Q_topredict))
        if len(X_topredict) == 0:
            print("WARNING: Data to predict is an empty list")
            return []
        X = parallel_sentences_to_bert_ids(X_topredict, self.sentence_length, self.tokenizer)
        Q = parallel_sentences_to_bert_ids(Q_topredict, self.sentence_length, self.tokenizer)
        X = np.array(X)
        Q = np.array(Q)
        return self.model.predict({'X':X, 'Q': Q, 'X2': np.array(X_positions, dtype=np.float32)})

    def test(self, X_test, Q_test, Y, X_positions=[]):
        assert(len(X_test) == len(Q_test))
        assert(len(X_test) == len(Y))

        X = parallel_sentences_to_bert_ids(X_test, self.sentence_length, self.tokenizer)
        Q = parallel_sentences_to_bert_ids(Q_test, self.sentence_length, self.tokenizer)
        return self.__test__(X,
                             Q,
                             Y,
                             X_positions)

class SimilaritiesBioBERT(SimilaritiesBERT):
    """A classifier that uses BioBERT for similarities
    This system uses the BioBERT weights converted for huggingface as described in 
    https://stackoverflow.com/questions/60539758/biobert-for-keras-version-of-huggingface-transformers"""
    def load_tokenizer(self):
        return BertTokenizer.from_pretrained('biobert_v1.1_pubmed', from_pt=True)

    def load_bert_model(self):
        return TFBertModel.from_pretrained('biobert_v1.1_pubmed', from_pt=True)

    def name(self):
        if self.siamese:
            str_siamese = "Siamese"
        else:
            str_siamese = ""
        if self.trainable:
            str_trainable = "Trainable"
        else:
            str_trainable = ""
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "BioBERT%s%sMean%s%s%s" % (str_siamese, str_trainable, self.comparison.name, str_positions, str_hidden)


class CNNSimilarities(Similarities):
    """A classifier that incorporates similarity operations"""
    NGRAMS = (2, 3, 4)
    CONVOLUTION = 32

    def name(self):
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "CNN%s%s%s" % (self.comparison.name, str_embeddings, str_hidden)

    def embedding_reduction(self, keep_prob):
        convs = []
        input_layer = keras.layers.Input(shape=(self.sentence_length, EMBEDDINGS))
        for ngram in self.NGRAMS:
            c = keras.layers.Conv1D(self.CONVOLUTION, ngram, activation='relu')(input_layer)
            m = keras.layers.GlobalMaxPooling1D()(c)
            convs.append(m)
        cnn_layer = keras.layers.Concatenate()(convs)
        return keras.models.Model(input_layer, cnn_layer)


class LSTMSimilarities(Similarities):
    """A classifier that incorporates similarity operations"""
    def name(self):
        if self.siamese:
            str_siamese = "Siamese"
        else:
            str_siamese = ""
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "BiLSTM%s%s%s%s%s" % (str_siamese, self.comparison.name, str_embeddings, str_positions, str_hidden)

    def embedding_reduction(self, keep_prob):
        input_layer = keras.layers.Input(shape=(self.sentence_length, EMBEDDINGS))
        embedding_layer = keras.layers.Bidirectional(keras.layers.LSTM(EMBEDDINGS, dropout=1-keep_prob))(input_layer)
        return keras.models.Model(input_layer, embedding_layer)


class LSTMSimilaritiesBERT(SimilaritiesBERT):
    """A classifier that incorporates similarity operations"""
    def name(self):
        if self.siamese:
            str_siamese = "Siamese"
        else:
            str_siamese = ""
        if self.trainable:
            str_trainable = "Trainable"
        else:
            str_trainable = ""
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "BERT%s%sBiLSTM%s%s%s" % (str_siamese, str_trainable, self.comparison.name, str_positions, str_hidden)

    def embedding_reduction(self, keep_prob):
        input_layer = keras.layers.Input(shape=(self.sentence_length, 768))
        embedding_layer = keras.layers.Bidirectional(keras.layers.LSTM(768, dropout=1-keep_prob))(input_layer)
        return keras.models.Model(input_layer, embedding_layer)

class LSTMSimilaritiesBioBERT(SimilaritiesBioBERT):
    """A classifier that incorporates similarity operations"""
    def name(self):
        if self.siamese:
            str_siamese = "Siamese"
        else:
            str_siamese = ""
        if self.trainable:
            str_trainable = "Trainable"
        else:
            str_trainable = ""
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "BioBERT%s%sBiLSTM%s%s%s" % (str_siamese, str_trainable, self.comparison.name, str_positions, str_hidden)

    def embedding_reduction(self, keep_prob):
        input_layer = keras.layers.Input(shape=(self.sentence_length, 768))
        embedding_layer = keras.layers.Bidirectional(keras.layers.LSTM(768, dropout=1-keep_prob))(input_layer)
        return keras.models.Model(input_layer, embedding_layer)

if __name__ == "__main__":
    import doctest
    import codecs
    doctest.testmod()

    # sys.exit()

    import csv

    def rouge_to_labels(rougeFile, labels, labelsthreshold, metric=["SU4"]):
        """Convert ROUGE values into classification labels
        >>> labels = rouge_to_labels("rouge_6b.csv", "topn", 3)
        Setting top 3 classification labels
        >>> labels[(0, '55031181e9bde69634000014', 9)]
        False
        >>> labels[(0, '55031181e9bde69634000014', 20)]
        True
        >>> labels[(0, '55031181e9bde69634000014', 3)]
        True
        """
        assert labels in ["topn", "threshold"]

        # Collect ROUGE values
        rouge = dict()
        with codecs.open(rougeFile,'r','utf-8') as f:
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
                    rouge[key] = np.mean([float(line[i]) for i in index])

        # Convert ROUGE values into classification labels
        result = dict()
        if labels == "threshold":
            print("Setting classification labels with threshold", labelsthreshold)
            for key, value in rouge.items():
                result[key] = (value >= labelsthreshold)
            qids = set(k[0] for k in rouge)
            # Regardless of threshold, set top ROUGE of every question to True
            for qid in qids:
                qid_items = [(key, value) for key, value in rouge.items() if key[0] == qid]
                qid_items.sort(key = lambda k: k[1])
                result[qid_items[-1][0]] = True
        else:
            print("Setting top", labelsthreshold, "classification labels")
            qids = set(k[0] for k in rouge)
            for qid in qids:
                qid_items = [(key, value) for key, value in rouge.items() if key[0] == qid]
                qid_items.sort(key = lambda k: k[1])
                for k, v in qid_items[-labelsthreshold:]:
                    result[k] = True
                for k, v in qid_items[:-labelsthreshold]:
                    result[k] = False
        return result

#    nnc = BasicNN(hidden_layer=50, build_model=True)
#   nnc = Similarities(hidden_layer=50, build_model=True, positions=True, siamese=True)
    nnc = LSTMSimilarities(hidden_layer=50, build_model=True, positions=True, siamese=True)
#    nnc = BasicBERT(hidden_layer=50, build_model=True)
#    nnc = SimilaritiesBERT(hidden_layer=50, build_model=True, positions=True, trainable=True, batch_size=8)



    labels_dict = rouge_to_labels('rouge_8b.csv', "topn", 5)
    sentences = []
    labels = []
    s_ids = []
    with open('rouge_8b.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences.append(nnc.cleantext(row['sentence text']))
            labels.append(labels_dict[(int(row['qid']), row['pubmedid'], int(row['sentid']))])
            s_ids.append([int(row['sentid'])])
    print("Data has %i items" % len(sentences))
    #print(sentences[:3])
    #print(labels[:3])

    print("Training %s" % nnc.name())
    loss = nnc.fit(sentences[100:500], 
                   sentences[100:500], 
                   np.array(labels[100:500]),
                   X_positions=np.array(s_ids[100:500]),
                   verbose=2,
                   validation_data=(sentences[:100], sentences[:100], labels[:100], s_ids[:100]),
                   nb_epoch=3)
    print("Training loss of each epoch: %s" % (str(loss['loss'])))
    print("Validation loss of each epoch: %s" % (str(loss['val_loss'])))
    testloss = nnc.test(sentences[:100], 
                        sentences[:100], 
                        np.array(labels[:100]),
                        X_positions=np.array(s_ids[:100]))
    print("Test loss: %s" % (testloss[0]))
    
