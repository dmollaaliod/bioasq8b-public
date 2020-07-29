# -*- coding: utf-8 -*-
"""regressionbaselines.py -- some regression baselines
Created on Tue Nov 15 18:18:08 2016

@author: diego
"""
import numpy as np
import scipy

#from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm

from nnmodels import simple, compare
from nnmodels.lstm import parallel_sentences_to_ids

class Constant():
    """A simple baseline that returns the mean target of the training data"""
    def name(self):
        return "Constant"

#    def fit(self, _X, _Q, Y, verbose=None,
#            nb_epoch=None, use_peepholes=None):
#        return self.__fit__(None, None, None, None, Y)

    def __fit__(self, _X, _sequence_lenths_x, 
                      _Q, _sequence_lengths_q,
                      Y, verbose=None, nb_epoch=None, use_peepholes=None,
                      learningrate=None, dropoutrate=None):
        self.mean = Y.mean()
        return np.mean((Y - self.mean)**2)

#    def predict(self, X_topredict, _Q_topredict):
#        return [[self.mean] for x in X_topredict]
    
#    def test(self, _X_test, _Q_test, Y):
#        return self.__test__(None, None, None, None, Y)
        
    def __test__(self, _X, _sequence_lengths_x, 
                       _Q, _sequence_lengths_q, Y):
        return np.mean((Y - self.mean)**2)

class BaseSVR:
    """A base class that uses SVR"""
    def __init__(self, kernel="rbf", C=1.0, gamma='auto'):
        self.kernel=kernel
        self.C=C
        self.gamma=gamma

    def __start__(self):
        self.regression = svm.SVR(kernel=self.kernel,
                                  C=self.C,
                                  gamma=self.gamma)
                                  
    def __yield_data__(self, X, sequence_lengths_x):
        for i, x in enumerate(X):
            #if i <= 10:
            #    print(i, sequence_lengths_x[i], x[:10])
            yield x[:sequence_lengths_x[i]]


class TfidfSVR(BaseSVR):
    """A baseline that uses SVR on tfidf of input sentences"""        
    def name(self):
        return "TfidfSVR"
                
    def __fit__(self, X, sequence_lengths_x, 
                      _Q, _sequence_lengths_q,
                      Y, verbose=None, nb_epoch=None, use_peepholes=None,
                      learningrate=None, dropoutrate=None):                    
        self.tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
        print("Generating training tfidf features")
        features = self.tfidf.fit_transform(self.__yield_data__(X, sequence_lengths_x))
        #print(len(self.tfidf.get_feature_names()), self.tfidf.get_feature_names()[:10])
        #print(features[0,0:10])
        print("Training tfidf features generated")
        print("Starting feature reduction")
        self.svd = TruncatedSVD(n_components=100)
        features_svd = self.svd.fit_transform(features)
        print("Feature reduction completed")
        self.__start__()
        #print(len(Y))
        print("Training SVR")
        self.regression.fit(features_svd, np.ravel(Y))
        print("SVR trained")
        predictions = self.regression.predict(features_svd)
        #print(mean_squared_error(Y, predictions))
        #print(np.mean((Y - predictions)**2))
        return mean_squared_error(Y, predictions)
        
    def __test__(self, X, sequence_lengths_x, 
                       _Q, _sequence_lengths_q, Y):
        features = self.tfidf.transform(self.__yield_data__(X, sequence_lengths_x))
        features_svd = self.svd.transform(features)
        predictions = self.regression.predict(features_svd)
        return mean_squared_error(Y, predictions)    

class TfidfNNR(BaseSVR):
    """A base class that uses NN"""
    def __init__(self, batch_size=128, n_components=100):
        self.batch_size = batch_size
        self.n_components = n_components

    def __start__(self):
        self.regression = simple.SingleNNR(batch_size=self.batch_size)

    def name(self):
        return "Tfidf%iNNR" % self.n_components

    def fit(self, X_train, _Q_train, Y_train,
            verbose=2, nb_epoch=3, use_peepholes=False,
            learningrate=0.001, dropoutrate=0.5, savepath=None, restore_model=False):
        assert(len(X_train) == len(Y_train))
        
        X, sequence_lengths_x = parallel_sentences_to_ids(X_train)
        sequence_lengths_x = np.array(sequence_lengths_x)

        return self.__fit__(X, sequence_lengths_x, None, None, Y_train,
                            verbose, nb_epoch, use_peepholes, learningrate, 
                            dropoutrate, savepath, restore_model)

    def __fit__(self, X, sequence_lengths_x, 
                      _Q, _sequence_lengths_q,
                      Y, verbose=1, nb_epoch=5, use_peepholes=None,
                      learningrate=0.001, dropoutrate=0.5, savepath=None, restore_model=False):                    
        self.tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
        print("Generating training tfidf features")
        features = self.tfidf.fit_transform(self.__yield_data__(X, sequence_lengths_x))
        #print(len(self.tfidf.get_feature_names()), self.tfidf.get_feature_names()[:10])
        #print(features[0,0:10])
        print("Training tfidf features generated")
        print(features.shape)
        if self.n_components == 0:
            features_svd = features
        else:
            print("Starting feature reduction")
            self.svd = TruncatedSVD(n_components=self.n_components)
            features_svd = self.svd.fit_transform(features)
            print("Feature reduction completed")
        print(features_svd.shape)
        self.__start__()
        if restore_model:
            self.regression.restore(savepath, features_svd.shape[1])
        else:
            print("Training NNR")
            self.regression.fit(features_svd, Y, 
                                verbose=verbose, 
                                learningrate=learningrate,
                                nb_epoch=nb_epoch,
                                dropoutrate=dropoutrate,
                                savepath=savepath)
            print("NNR trained")
        predictions = self.regression.predict(features_svd)
        #print(mean_squared_error(Y, predictions))
        #print(np.mean((Y - predictions)**2))
        return mean_squared_error(Y, predictions)

    def test(self, X_test, _Q_test, Y):
        assert(len(X_test) == len(Y))

        X, sequence_lengths_x = parallel_sentences_to_ids(X_test)
        return self.__test__(X, sequence_lengths_x,
                             None, None,
                             Y)
        
    def __test__(self, X, sequence_lengths_x, 
                       _Q, _sequence_lengths_q, Y):
        features = self.tfidf.transform(self.__yield_data__(X, sequence_lengths_x))
        if self.n_components == 0:
            features_svd = features
        else:
            features_svd = self.svd.transform(features)
        predictions = self.regression.predict(features_svd)
        return mean_squared_error(Y, predictions)    

    def predict(self, X_topredict, _Q_topredict):
        X, sequence_lengths_x = parallel_sentences_to_ids(X_topredict)
        features = self.tfidf.transform(self.__yield_data__(X, sequence_lengths_x))
        if self.n_components == 0:
            features_svd = features
        else:
            features_svd = self.svd.transform(features)
        predictions = self.regression.predict(features_svd)
        return predictions


class TfidfSimSVR(BaseSVR):
    """A baseline that uses SVR on tfidf of input sentences plus distance"""        
    def name(self):
        return "TfidfSimSVR"
                
    def __fit__(self, X, sequence_lengths_x, 
                      Q, sequence_lengths_q,
                      Y, verbose=None, nb_epoch=None, use_peepholes=None,
                      learningrate=None, dropoutrate=None):  
        assert len(X) == len(Q)
        assert len(X) == len(Y)
        self.tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
        print("Generating training tfidf features")
        features = self.tfidf.fit_transform(self.__yield_data__(X, sequence_lengths_x))
        features_q = self.tfidf.transform(self.__yield_data__(Q, sequence_lengths_q))
        #print(len(self.tfidf.get_feature_names()), self.tfidf.get_feature_names()[:10])
        #print(features[0,0:10])
        print("Training tfidf features generated")
        print("Starting feature reduction")
        self.svd = TruncatedSVD(n_components=5)
        features_svd = self.svd.fit_transform(features)
        features_svd_q = self.svd.transform(features_q)
        print("Feature reduction completed")
        print("Computing distances")
        distances = [cosine_similarity([features_svd[i,:]], [features_svd_q[i,:]])[0]
                     for i in range(len(X))]
        all_features = np.hstack((features_svd, distances))
        print(all_features.shape)                 
        self.__start__()
        #print(len(Y))
        print("Training SVR")
        self.regression.fit(all_features, np.ravel(Y))
        print("SVR trained")
        predictions = self.regression.predict(all_features)
        #print(mean_squared_error(np.ravel(Y), predictions))
        #print(np.mean((np.ravel(Y) - predictions)**2))
        #print(np.ravel(Y).shape, predictions.shape)
        return mean_squared_error(Y, predictions)
        
    def __test__(self, X, sequence_lengths_x, 
                       Q, sequence_lengths_q, Y):
        features = self.tfidf.transform(self.__yield_data__(X, sequence_lengths_x))
        features_q = self.tfidf.transform(self.__yield_data__(Q, sequence_lengths_q))
        features_svd = self.svd.transform(features)
        features_svd_q = self.svd.transform(features_q)
        distances = [cosine_similarity([features_svd[i,:]], [features_svd_q[i,:]])[0]
                     for i in range(len(X))]
        all_features = np.hstack((features_svd, distances))                 
        predictions = self.regression.predict(all_features)
        return mean_squared_error(Y, predictions)

class TfidfSimNNR(BaseSVR):
    """A baseline that uses NN on tfidf of input sentences plus distance"""        
    def __init__(self, batch_size=128, n_components=100):
        self.batch_size = batch_size
        self.n_components = n_components

    def __start__(self):
        self.regression = simple.SingleNNR(batch_size=self.batch_size)

    def name(self):
        return "Tfidf%iSimNNR" % self.n_components

    def fit(self, X_train, Q_train, Y_train,
            verbose=2, nb_epoch=3, use_peepholes=False,
            learningrate=0.001, dropoutrate=0.5, savepath=None, restore_model=False):
                
        assert len(X_train) == len(Q_train)
        assert len(X_train) == len(Y_train)
        
        X, sequence_lengths_x = parallel_sentences_to_ids(X_train)
        sequence_lengths_x = np.array(sequence_lengths_x)
        Q, sequence_lengths_q = parallel_sentences_to_ids(Q_train)
        sequence_lengths_q = np.array(sequence_lengths_q)

        return self.__fit__(X, sequence_lengths_x, Q, sequence_lengths_q, Y_train,
                            verbose=verbose, nb_epoch=nb_epoch, 
                            use_peepholes=use_peepholes, 
                            learningrate=learningrate, dropoutrate=dropoutrate,
                            savepath=savepath, restore_model=restore_model)
                
    def __fit__(self, X, sequence_lengths_x, 
                      Q, sequence_lengths_q,
                      Y, verbose=1, nb_epoch=5, use_peepholes=None,
                      learningrate=0.001, dropoutrate=0.5, savepath=None,
                      restore_model=False): 
        assert len(X) == len(Q)
        assert len(X) == len(Y)
        self.tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
        print("Generating training tfidf features")
        features = self.tfidf.fit_transform(self.__yield_data__(X, sequence_lengths_x))
        features_q = self.tfidf.transform(self.__yield_data__(Q, sequence_lengths_q))
        #print(len(self.tfidf.get_feature_names()), self.tfidf.get_feature_names()[:10])
        #print(features[0,0:10])
        print("Training tfidf features generated")
        if self.n_components == 0:
            features_svd = features
            features_svd_q = features_q
            print("Computing distances")
            distances = [cosine_similarity(features_svd[i,:].todense(), features_svd_q[i,:].todense())[0]
                         for i in range(len(X))]
            all_features = scipy.sparse.hstack((features_svd, distances)).tocsr()
        else:
            print("Starting feature reduction")
            self.svd = TruncatedSVD(n_components=self.n_components)
            features_svd = self.svd.fit_transform(features)
            features_svd_q = self.svd.transform(features_q)
            print("Feature reduction completed")
            print("Computing distances")
            distances = [cosine_similarity([features_svd[i,:]], [features_svd_q[i,:]])[0]
                         for i in range(len(X))]
            all_features = np.hstack((features_svd, distances))
        print(all_features.shape)                 
        self.__start__()
        if restore_model:
            self.regression.restore(savepath, all_features.shape[1])
        else:
            print("Training NNR")
            self.regression.fit(all_features, Y,
                                verbose=verbose, 
                                learningrate=learningrate,
                                nb_epoch=nb_epoch,
                                dropoutrate=dropoutrate,
                                savepath=savepath)
            print("NNR trained")
        predictions = self.regression.predict(all_features)
        #print(mean_squared_error(np.ravel(Y), predictions))
        #print(np.mean((np.ravel(Y) - predictions)**2))
        #print(np.ravel(Y).shape, predictions.shape)
        return mean_squared_error(Y, predictions)
        
    def test(self, X_test, Q_test, Y):
        assert(len(X_test) == len(Q_test))
        assert(len(X_test) == len(Y))

        X, sequence_lengths_x = parallel_sentences_to_ids(X_test)
        Q, sequence_lengths_q = parallel_sentences_to_ids(Q_test)
        return self.__test__(X, sequence_lengths_x,
                             Q, sequence_lengths_q,
                             Y)

    def __test__(self, X, sequence_lengths_x, 
                       Q, sequence_lengths_q, Y):
        features = self.tfidf.transform(self.__yield_data__(X, sequence_lengths_x))
        features_q = self.tfidf.transform(self.__yield_data__(Q, sequence_lengths_q))
        if self.n_components == 0:
            features_svd = features
            features_svd_q = features_q
            distances = [cosine_similarity(features_svd[i,:].todense(), features_svd_q[i,:].todense())[0]
                         for i in range(len(X))]
            all_features = scipy.sparse.hstack((features_svd, distances)).tocsr()                 
        else:    
            features_svd = self.svd.transform(features)
            features_svd_q = self.svd.transform(features_q)
            distances = [cosine_similarity([features_svd[i,:]], [features_svd_q[i,:]])[0]
                         for i in range(len(X))]
            all_features = np.hstack((features_svd, distances))                 
        predictions = self.regression.predict(all_features)
        return mean_squared_error(Y, predictions)

    def predict(self, X_topredict, Q_topredict):
        assert(len(X_topredict) == len(Q_topredict))
        X, sequence_lengths_x = parallel_sentences_to_ids(X_topredict)
        Q, sequence_lengths_q = parallel_sentences_to_ids(Q_topredict)
        features = self.tfidf.transform(self.__yield_data__(X, sequence_lengths_x))
        features_q = self.tfidf.transform(self.__yield_data__(Q, sequence_lengths_q))
        if self.n_components == 0:
            features_svd = features
            features_svd_q = features_q
            distances = [cosine_similarity(features_svd[i,:].todense(), features_svd_q[i,:].todense())[0]
                         for i in range(len(X))]
            all_features = scipy.sparse.hstack((features_svd, distances)).tocsr()                 
        else:
            features_svd = self.svd.transform(features)
            features_svd_q = self.svd.transform(features_q)
            distances = [cosine_similarity([features_svd[i,:]], [features_svd_q[i,:]])[0]
                         for i in range(len(X))]
            all_features = np.hstack((features_svd, distances))                 
        predictions = self.regression.predict(all_features)
        return predictions

class TfidfSim2NNR(TfidfSimNNR):
    """A baseline that integrates the similarity in the NNR model"""
    def __init__(self, batch_size=128, n_components=100, comparison=compare.SimMul()):
        self.batch_size = batch_size
        self.n_components = n_components
        self.comparison = comparison

    def __start__(self):
        self.regression = simple.SimNNR(batch_size=self.batch_size,
                                        comparison=self.comparison)

    def name(self):
        return "Tfidf%i-%s-relu" % (self.n_components, self.comparison.name)
                         
    def __fit__(self, X, sequence_lengths_x, 
                      Q, sequence_lengths_q,
                      Y, verbose=1, nb_epoch=5, use_peepholes=None,
                      learningrate=0.001, dropoutrate=0.5, savepath=None,
                      restore_model=False):  
        assert len(X) == len(Q)
        assert len(X) == len(Y)
        self.tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
        print("Generating training tfidf features")
        features = self.tfidf.fit_transform(self.__yield_data__(X, sequence_lengths_x))
        features_q = self.tfidf.transform(self.__yield_data__(Q, sequence_lengths_q))
        print("Training tfidf features generated")
        if self.n_components == 0:
            features_svd = features
            features_svd_q = features_q
        else:
            print("Starting feature reduction")
            self.svd = TruncatedSVD(n_components=self.n_components)
            features_svd = self.svd.fit_transform(features)
            features_svd_q = self.svd.transform(features_q)
            print("Feature reduction completed")
        self.__start__()
        if restore_model:
            self.regression.restore(savepath, self.n_components)
        else:
            print("Training NNR")
            self.regression.fit(features_svd, features_svd_q, Y,
                                verbose=verbose, 
                                learningrate=learningrate,
                                nb_epoch=nb_epoch,
                                dropoutrate=dropoutrate,
                                savepath=savepath)
            print("NNR trained")
        predictions = self.regression.predict(features_svd, features_svd_q)
        #print(mean_squared_error(np.ravel(Y), predictions))
        #print(np.mean((np.ravel(Y) - predictions)**2))
        #print(np.ravel(Y).shape, predictions.shape)
        return mean_squared_error(Y, predictions)
        
    def __test__(self, X, sequence_lengths_x, 
                       Q, sequence_lengths_q, Y):
        features = self.tfidf.transform(self.__yield_data__(X, sequence_lengths_x))
        features_q = self.tfidf.transform(self.__yield_data__(Q, sequence_lengths_q))
        if self.n_components == 0:
            features_svd = features
            features_svd_q = features_q
        else:
            features_svd = self.svd.transform(features)
            features_svd_q = self.svd.transform(features_q)
        predictions = self.regression.predict(features_svd, features_svd_q)
        return mean_squared_error(Y, predictions)

    def predict(self, X_topredict, Q_topredict):
        assert(len(X_topredict) == len(Q_topredict))
        X, sequence_lengths_x = parallel_sentences_to_ids(X_topredict)
        Q, sequence_lengths_q = parallel_sentences_to_ids(Q_topredict)
        features = self.tfidf.transform(self.__yield_data__(X, sequence_lengths_x))
        features_q = self.tfidf.transform(self.__yield_data__(Q, sequence_lengths_q))
        if self.n_components == 0:
            features_svd = features
            features_svd_q = features_q
        else:
            features_svd = self.svd.transform(features)
            features_svd_q = self.svd.transform(features_q)
        predictions = self.regression.predict(features_svd, features_svd_q)
        return predictions
