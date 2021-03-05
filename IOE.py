import copy as cp
import warnings
import numpy as np
from imblearn.over_sampling import SMOTE
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.drift_detection import ADWIN
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.utils import check_random_state
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.utils.utils import *
import itertools
import random




class IOE_Classifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Improved Online ensemble classifier.
    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=KNNADWINClassifier)
        Each member of the ensemble is an instance of the base estimator.
    m: int (default=10)
        The size of the ensemble, in other words, how many classifiers to train.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
    Threshold: float, threshold used for balancing the classes
    forgetting_factor: float, Forgetting factor for calculating the time-decaying recals and size for classes. 

    Raises
    ------
    ValueError: A ValueError is raised if the 'classes' parameter is
    not passed in the first partial_fit call.
    Notes
    -----


    """

    def __init__(self, base_estimator=HoeffdingTreeClassifier(), random_state=None,  threshold=0.05,
                  m=10,forgetting_factor = 0.9):
        super().__init__()
        # default values
        self.ensemble = None


        self.actual_n_estimators = None
        self.classes = None
        self.threshold = threshold
        self._random_state = None  # This is the actual random_state object used internally
        self.base_estimator = base_estimator
        self.m = m
        self.random_state = random_state
        self.w = {}
        self.recalls = {}
        self.forgetting_factor = forgetting_factor
        self.first_call = True
        self.__configure()

    def __configure(self):
        if hasattr(self.base_estimator, "reset"):
            self.base_estimator.reset()
        self.actual_n_estimators = self.m
        self.ensemble = [cp.deepcopy(self.base_estimator) for _ in range(self.actual_n_estimators)]
        self._random_state = check_random_state(self.random_state)

    def reset(self):
        self.__configure()
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None,warm_up = False):
        """
        Partially (incrementally) fit the model.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.
        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.
        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.
        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending
            on the base estimator.
        Raises
        ------
        ValueError
            A ValueError is raised if the 'classes' parameter is not passed in the first
            partial_fit call, or if they are passed in further calls but differ from
            the initial classes list passed.

        Returns
        -------
        OzaBaggingClassifier
            self
        Notes
        -----
        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.
        Each sample is trained by each classifier a total of K times, where K
        is drawn by a Poisson(1) distribution.
        """

        if self.classes is None:
            if classes is None:
                raise ValueError("The first partial_fit call should pass all the classes.")
            else:
                self.classes = classes
                for cl in self.classes:
                    self.w[cl] = 0
                    self.recalls[cl] = 0


        if self.classes is not None and classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError("The classes passed to the partial_fit function differ from those passed earlier.")


        L_classes = {}
        for cl in self.classes:
            L_classes[cl] = 1
        # if w is not None:
        values = self.w.values()
        nb_of_zeros_w = np.count_nonzero(np.array(list(self.w.values())) == 0.0)
        nb_of_zeros = np.count_nonzero(np.array(list(self.recalls.values())) == 0.0)

        if nb_of_zeros != len(self.recalls) and nb_of_zeros_w != len(self.w):
            values = self.w.values()
            sum_values = sum(values)
            w_mean = sum_values / len(self.w)
            avg_recalls = sum(self.recalls.values()) / float(len(self.recalls))
            for var in self.classes:
                if abs(self.recalls[var] - avg_recalls) > self.threshold:
                    if self.w[var] == 0:
                        self.w[var] = 1



                    if self.w[var] == 0:
                        self.w[var] = 1
                    if (self.recalls[var] != 0.0 and self.recalls[var] < avg_recalls and self.w[var] < w_mean):
                        L_classes[var] = (((sum(self.recalls.values()) - self.recalls[var]) / float(len(self.recalls) - 1)) /
                                               self.recalls[var]) * (max(self.w.values())) / self.w[var]

                    elif (self.recalls[var] != 0.0):
                        L_classes[var] = (
                                    ((sum(self.recalls.values()) - self.recalls[var]) / float(len(self.recalls) - 1)) / self.recalls[var])
                    else:
                        L_classes[var] = 10
                    if L_classes[var] > 10:
                        L_classes[var] = 10


        self.__adjust_ensemble_size()

        r, _ = get_dimensions(X)

        for j in range(r):
            if not self.first_call:

                result = self.predict([X[j]])
                if result[0] == y[j]:
                    self.recalls[result[0]] = self.forgetting_factor * self.recalls[result[0]] + (
                                1 - self.forgetting_factor)
                else:
                    self.recalls[y[j]] = self.forgetting_factor * self.recalls[y[j]]
                for cl in self.classes:
                    if cl == y[j]:
                        self.w[y[j]] = self.forgetting_factor * self.w[y[j]] + (
                                1 - self.forgetting_factor)
                    else:
                        self.w[cl] = self.forgetting_factor * self.w[cl]
            else:
                self.first_call = False
            for i in range(self.actual_n_estimators):
                weight = 1
                k = self._random_state.poisson(L_classes[y[j]])
                weight = L_classes[y[j]]


                if k > 0:
                    for b in range(k):

                        self.ensemble[i].partial_fit([X[j]], [y[j]], self.classes, [weight])

        return self

    def __adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.base_estimator))
                    self.actual_n_estimators += 1

    def predict(self, X):
        """ Predict classes for the passed electricity.csv.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of electricity.csv samples to predict the class labels for.
        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.
        Notes
        -----
        The predict function will average the predictions from all its learners
        to find the most likely prediction for the sample matrix X.
        """
        r, c = get_dimensions(X)
        proba = self.predict_proba(X)
        # print(X)
        # print(proba)
        predictions = []
        if proba is None:
            return None
        for i in range(r):
            predictions.append(np.argmax(proba[i]))
        # print(predictions)
        return np.asarray(predictions)

    def predict_proba(self, X):
        """ Estimates the probability of each sample in X belonging to each of the class-labels.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.
        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated with the X entry of the
        same index. And where the list in index [i] contains len(self.target_values) elements, each of which represents
        the probability that the i-th sample of X belongs to a certain class-label.

        Raises
        ------
        ValueError: A ValueError is raised if the number of classes in the base_estimator
        learner differs from that of the ensemble learner.
        """
        proba = []
        r, c = get_dimensions(X)
        try:
            for i in range(self.actual_n_estimators):
                partial_proba = self.ensemble[i].predict_proba(X)
                if len(partial_proba[0]) > max(self.classes) + 1:
                    raise ValueError("The number of classes in the base learner is larger than in the ensemble.")

                if len(proba) < 1:
                    for n in range(r):
                        proba.append([0.0 for _ in partial_proba[n]])

                for n in range(r):
                    for l in range(len(partial_proba[n])):
                        try:
                            proba[n][l] += partial_proba[n][l]
                        except IndexError:
                            proba[n].append(partial_proba[n][l])
        except ValueError:
            return np.zeros((r, 1))
        except TypeError:
            return np.zeros((r, 1))

        # normalizing probabilities
        sum_proba = []
        for l in range(r):
            sum_proba.append(np.sum(proba[l]))
        aux = []
        for i in range(len(proba)):
            if sum_proba[i] > 0.:
                aux.append([x / sum_proba[i] for x in proba[i]])
            else:
                aux.append(proba[i])
        return np.asarray(aux)