import logging
from scipy import sparse
import numpy as np
from spotad.data.common import AbstractMultiLabelDataGroup

mlds_logger = logging.getLogger("SciPyMultiLabelDataSet")


class SciPyMultiLabelDataGroup(AbstractMultiLabelDataGroup):
    def __init__(self, named_multi_sets):
        super(SciPyMultiLabelDataGroup, self).__init__(named_multi_sets)

    def focus_on_set(self, label):
        return SciPyDataGroup({name: mlds.focus_on_set(label) for name, mlds in self._named_multi_sets.iteritems()})


class SciPyDataGroup(object):
    def __init__(self, named_sets):
        self._named_sets = named_sets
        self._in_context = False

    def __getitem__(self, name):
        return self._named_sets[name]

    @property
    def names(self):
        return self._named_sets.keys()

    def __enter__(self):
        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_context = False

    def unified_set(self):
        return SciPyCombinedDataSet(self)


class SciPyCombinedDataSet(object):
    def __init__(self, data_group):
        self._in_context = False
        self._X = None
        self._y = None
        self._header = None
        self._data_group = data_group

    def _validate_in_context(self, prop_name):
        if not self._in_context:
            raise AssertionError("property '{}' accessed outside a with block".format(prop_name))

    @property
    def header(self):
        self._validate_in_context('header')
        return self._header

    @property
    def X(self):
        self._validate_in_context('X')
        return self._X

    @property
    def y(self):
        self._validate_in_context('y')
        return self._y

    def __enter__(self):
        # ASSUMPTION: all data sets in the group have the same header
        self._header = self._data_group[self._data_group.names[0]].header
        self._X = sparse.vstack(tuple([self._data_group[n].X for n in self._data_group.names]))
        self._y = np.hstack(tuple([self._data_group[n].y for n in self._data_group.names]))

        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._header = None
        self._X = None
        self._y = None
        self._in_context = False