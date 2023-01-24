from spotad.data.common import AbstractMultiLabelDataGroup
import xgboost as xgb
import numpy as np
from scipy import sparse


class XGBoostMultiLabelDataGroup(AbstractMultiLabelDataGroup):
    def __init__(self, named_multi_sets):
        super(XGBoostMultiLabelDataGroup, self).__init__(named_multi_sets)

    def focus_on_set(self, label):
        return XGBoostDataGroup({name: mlds.focus_on_set(label) for name, mlds in self._named_multi_sets.iteritems()})


class XGBoostDataGroup(object):
    def __init__(self, named_sets):
        self._named_sets = named_sets
        self._realized_named_sets = dict()

    def __getitem__(self, name):
        return self._realized_named_sets[name]

    @property
    def names(self):
        return self._named_sets.keys()

    def __enter__(self):
        self._in_context = True
        self._realized_named_sets = dict()

        for name in self.names:
            self._realized_named_sets[name] = XGBoostDataSet(self._named_sets[name])

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_context = False

    def unified_set(self):
        return XGBBoostCombinedDataSet(self)


class XGBoostDataSet(object):
    def __init__(self, ds):
        self._header = ds.header + ['dummy_for_xgb']
        self._data = xgb.DMatrix(data=sparse.hstack((ds.X, np.ones((ds.X.shape[0], 1), dtype=np.uint8))),
                                 label=ds.y, feature_names=self._header)

    @property
    def header(self):
        return self._header

    @property
    def data_matrix(self):
        return self._data

    @property
    def gold_labels(self):
        return self._data.get_label()

    @property
    def y(self):
        return self.gold_labels


class XGBBoostCombinedDataSet(object):
    def __init__(self, data_group):
        self._in_context = False
        self._data = None
        self._data_group = data_group

    def _validate_in_context(self, prop_name):
        if not self._in_context:
            raise AssertionError("property '{}' accessed outside a with block".format(prop_name))

    @property
    def header(self):
        self._validate_in_context('header')
        return self._header

    @property
    def data_matrix(self):
        self._validate_in_context('data_matrix')
        return self._data

    def __enter__(self):
        # ASSUMPTION: all data sets in the group have the same header
        # ASSUMPTION2: header already contains the 'dummy_fo_xgb' so no need to add it again
        self._header = self._data_group[self._data_group.names[0]].header
        X = sparse.vstack(tuple([ sparse.hstack((self._data_group._named_sets[n].X,
                                                 np.ones((self._data_group._named_sets[n].X.shape[0], 1),
                                                         dtype=np.uint8)))
                                 for n in self._data_group.names]))
        y = np.hstack(tuple([self._data_group._named_sets[n].y for n in self._data_group.names]))

        self._data = xgb.DMatrix(data=X, label=y, feature_names=self._header)

        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._header = None
        self._data = None
        self._in_context = False
