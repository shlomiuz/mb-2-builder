from collections import namedtuple
import logging
import numpy as np

SingleLabelDataSet = namedtuple('SingleLabelDataSet', ['header', 'X', 'y'])


class MultiLabelDataSet(object):

    def __init__(self, header, X, ys):
        self._header = header
        self._X = X
        self._ys = ys

    @property
    def header(self):
        return self._header

    @property
    def X(self):
        return self._X

    @property
    def ys(self):
        return self._ys

    def focus_on_set(self, label):
        return SingleLabelDataSet(self._header, self._X, self._ys[label])


mlds_logger = logging.getLogger("AbstractMultiLabelDataGroup")
class AbstractMultiLabelDataGroup(object):

    def __init__(self, named_multi_sets):
        self._named_multi_sets = named_multi_sets

    def __getitem__(self, name):
        return self._named_multi_sets[name]

    @property
    def names(self):
        return self._named_multi_sets.keys()

    def filter_out_invalid_labels(self, goals):
        valid_goals = []

        def is_enough_diversity(y):
            # validate enough data exists
            label_counts = np.bincount(y.reshape(y.shape[0]))
            if len(label_counts) < 2 or label_counts[0] < 10 or label_counts[1] < 10:  # 10 - arbitrary small value
                return False
            else:
                return True

        for goal in goals:
            if all(map(lambda mlds: is_enough_diversity(mlds.ys[goal]), self._named_multi_sets.values())):
                valid_goals.append(goal)
            else:
                mlds_logger.warn("Not enough data available for goal: %s, Skipping", goal)
        return valid_goals

    def focus_on_set(self, label):
        raise NotImplemented("must be implemented by children")
