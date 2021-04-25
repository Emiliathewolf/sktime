# -*- coding: utf-8 -*-
"""
    Base Time Series Forest Class.
    An implementation of Deng's Time Series Forest, with minor changes.
"""

__author__ = ["Tony Bagnall", "kkoziara", "luiszugasti", "kanand77"]
__all__ = [
    "BaseTimeSeriesForest",
    "_transform",
    "_get_intervals",
    "_fit_estimator",
    "_predict_proba_for_estimator",
]

import math

import numpy as np
from joblib import Parallel
from joblib import delayed
from sklearn.base import clone
from sklearn.utils.multiclass import class_distribution
from sklearn.utils.validation import check_random_state

from scipy.spatial.distance import pdist

from sktime.utils.slope_and_trend import _slope
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y
from sktime.utils.validation.panel import check_consistent_length


class BaseTimeSeriesForest:
    """Base Time series forest classifier."""

    # Capabilities: data types this classifier can handle
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
    }

    def __init__(
        self,
        min_interval=3,
        n_estimators=200,
        n_jobs=8,
        random_state=None,
    ):
        super(BaseTimeSeriesForest, self).__init__(
            base_estimator=self._base_estimator,
            n_estimators=n_estimators,
        )

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        self.n_jobs = n_jobs
        # The following set in method fit
        self.n_classes = 0
        self.series_length = 0
        self.n_intervals = 0
        self.estimators_ = []
        self.intervals_ = []
        self.classes_ = []

        #Handling unequal length series
        self.unequal = False

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y) using random
        intervals and summary features
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances,
        series_length] or shape = [n_instances,n_columns]
            The training input samples.  If a Pandas data frame is passed it
            must have a single column (i.e. univariate
            classification. TSF has no bespoke method for multivariate
            classification as yet.
        y : array-like, shape =  [n_instances]    The class labels.
        unequal : bool
            Flag to adjust the fitting to account for unequal length series

        Returns
        -------
        self : object
        """
        if X.size < self.n_estimators:
            self.n_estimators = X.size

        #Try except for now
        try:
            X, y = check_X_y(
                X,
                y,
                enforce_univariate=not self.capabilities["multivariate"],
                coerce_to_numpy=True,
            )
            X = X.squeeze(1)
            #Number of instances, length of each series
            n_instances, self.series_length = X.shape



            rng = check_random_state(self.random_state)

            self.n_classes = np.unique(y).shape[0]

            self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
            self.n_intervals = int(math.sqrt(self.series_length))
            if self.n_intervals == 0:
                self.n_intervals = 1
            if self.series_length < self.min_interval:
                self.min_interval = self.series_length

            self.intervals_ = [
                _get_intervals(self.n_intervals, self.min_interval, self.series_length, rng)
                for _ in range(self.n_estimators)
            ]
            self.unequal = False

        # CHANGE THIS TO HANDLE SPECIFIC ERROR
        except ValueError:
            self.unequal = True
            n_instances = X.shape[0]
            rng = check_random_state(self.random_state)
            self.n_classes = np.unique(y).shape[0]
            self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

            self.intervals_ = []
            for i in range(self.n_estimators):
                series_length = X.iloc[i][0].size
                #print("Length X at index 100: ", series_length)
                n_intervals = int(math.sqrt(series_length))
                if n_intervals == 0:
                    n_intervals = 1
                if series_length < self.min_interval:
                    self.min_interval = series_length
                self.intervals_.append(_get_intervals(n_intervals, self.min_interval, series_length, rng))
            # Adjust it to take series length for each individual series

            #Lets try get away with sampling each array seperatly, and not in bulk
            # e.g. series length for each and every time series in teh file, not just once
        #Adjust for unequal length
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                X,
                y,
                self.base_estimator,
                self.intervals_[i],
                self.random_state,
                self.unequal
            )
            for i in range(self.n_estimators)
        )
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Find predictions for all cases in X. Built on top of predict_proba
        Parameters
        ----------
        X : The training input samples. array-like or pandas data frame.
        If a Pandas data frame is passed, a check is performed that it only
        has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        proba = self.predict_proba(X)
        return np.asarray([self.classes_[np.argmax(prob)] for prob in proba])

    def predict_proba(self, X):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances, series_length]
            If a Pandas data frame is passed (sktime format) a check is
            performed that it only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : nd.array of shape = (n_instances, n_classes)
            Predicted probabilities
        """
        self.check_is_fitted()
        if self.unequal == False:
            X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
            X = X.squeeze(1)
            _, series_length = X.shape
            if series_length != self.series_length:
                raise TypeError(
                    " ERROR number of attributes in the train does not match "
                    "that in the test data"
                )

        y_probas = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_proba_for_estimator)(
                X, self.estimators_[i], self.intervals_[i], self.unequal
            )
            for i in range(self.n_estimators)
        )
        np.ones(self.n_classes) * self.n_estimators
        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes) * self.n_estimators
        )
        #output = np.sum(y_probas, axis=0) / (
        #    np.ones(self.n_classes) * self.n_estimators
        #)
        return output


def us(query, scale_to, min = 0, max = None):
    n = len(query)
    m = scale_to

    QP = []
    # p / n = scaling factor
    if max is None:
        max = m
    for j in range(min, max):
        QP.append(query[int(j*(n/m))])
    return QP

def _transform(X, intervals, unequal):
    """Compute the mean, standard deviation and slope for given intervals
    of input data X.

    Args:
        X (Array-like, int or float): Time series data X
        intervals (Array-like, int or float): Time range intervals for series X

    Returns:
        int32 Array: transformed_x containing mean, std_deviation and slope
    """
    n_instances, _ = X.shape
    n_intervals, _ = intervals.shape
    transformed_x = np.empty(shape=(3 * n_intervals, n_instances), dtype=np.float32)
    for j in range(n_intervals):
        #Please set multicore before you do this, otherwise it will take 2+ minutes
        if unequal == True:
            best_match_value = float('inf')
            best_length = None

            #We don't want to waste time checking out all sizes, so set some upper and lower bounds for ourselves
            min_length = int(intervals[j][1]) # start
            max_length = int((min_length * 1.1) + 1) # up to 10% over as any higher is unlikely to be good match


            #work out ratio here to avoid nabbing the end all the time
            #longest length is
            longest_length = 0
            for i in range(X.size):
                if X.iloc[i][0].size > longest_length:
                    longest_length = X.iloc[i][0].size

            #ratio working out
            low_diff = intervals[j][0] / longest_length

            best_X_scale = []


            # LB_Keogh = sqrt(sum([[Q > U].* [Q-U]; [Q < L].* [L-Q]].^2));
            #C = X
            # n query length, m candidate length

            # We want the scale length to be greater than the min length to ensure we're not always picking the end
            for s in range(min_length, int(min_length + 1)):
                # Scale
                scaled = np.empty(shape=(int(n_instances), int(intervals[j][1] - intervals[j][0])))
                curX = None
                lenX = None

                lower_interval = int(s * low_diff)
                upper_interval = int(lower_interval + (intervals[j][1] - intervals[j][0]))
                # As we don't scale the interval size itself, sometimes it may run over
                if upper_interval > longest_length:
                    upper_interval = upper_interval - (upper_interval - longest_length) - 1
                    lower_interval = lower_interval - (upper_interval - longest_length) - 1
                for i in range(n_instances):
                    # Append the scaled and sliced result
                    curX = X.iloc[i][0]
                    lenX = len(curX)
                    for o in range(lower_interval, upper_interval):
                        scaled[i] = (curX[int(o * (lenX / s))])

                ED = sum(pdist(np.array(scaled), 'sqeuclidean'))
                if ED < best_match_value:
                    best_match_value = ED
                    best_X_scale = scaled

            X_slice = best_X_scale
            X_slice = np.array(X_slice)
        else:
            X_slice = X[:, intervals[j][0] : intervals[j][1]]

        try:
            means = np.mean(X_slice, axis=1)
        except:
            print("test")
        std_dev = np.std(X_slice, axis=1)
        slope = _slope(X_slice, axis=1)
        transformed_x[3 * j] = means
        transformed_x[3 * j + 1] = std_dev
        transformed_x[3 * j + 2] = slope
    return transformed_x.T


def _get_intervals(n_intervals, min_interval, series_length, rng):
    """
    Generate random intervals for given parameters.
    """
    intervals = np.zeros((n_intervals, 2), dtype=int)
    for j in range(n_intervals):
        intervals[j][0] = rng.randint(series_length - min_interval)
        length = rng.randint(series_length - intervals[j][0] - 1)
        if length < min_interval:
            length = min_interval
        intervals[j][1] = intervals[j][0] + length
    return intervals



def _fit_estimator(X, y, base_estimator, intervals, random_state=None, unequal = False):
    """
    Fit an estimator - a clone of base_estimator - on input data (X, y)
    transformed using the randomly generated intervals.
    """

    estimator = clone(base_estimator)
    estimator.set_params(random_state=random_state)

    transformed_x = _transform(X, intervals, unequal)
    return estimator.fit(transformed_x, y)


def _predict_proba_for_estimator(X, estimator, intervals, unequal):
    """
    Find probability estimates for each class for all cases in X using
    given estimator and intervals.
    """
    transformed_x = _transform(X, intervals, unequal)
    return estimator.predict_proba(transformed_x)
