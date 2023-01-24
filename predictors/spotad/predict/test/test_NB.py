from nose.tools import *
from pandas.util.testing import assert_contains_all

from spotad.predict.NB import BayesModel, ProbabilityToCRAdapter
import os

test_folder = os.path.dirname(__file__)


def test_convert_idexes():
    model = BayesModel(test_folder + '/simple.zip', 'historical')
    indexes = model._features_to_combination_indices(['a', 'c'])
    assert_contains_all([(0, 2)], indexes)
    assert_equal(len(indexes), 3)



def test_unknown_features_are_ignored():
    model = BayesModel(test_folder + '/simple.zip', 'historical')
    p1, _ = model.predict(['a', 'b'])
    p2, _ = model.predict(['a', 'b', 'g', 'f'])

    assert_equal(p1, p2)

def test_adapter_fallback():
    adapter = ProbabilityToCRAdapter("bla bla")
    assert_equal(adapter(0.3), 0.3)
    assert_equal(adapter(17), 17)


def test_adapter():
    # The file contains these settings
    # thresholds	3	2	1
    # crs	8	7	9	13
    adapter = ProbabilityToCRAdapter(test_folder + '/thresholds.zip')
    assert_equal(adapter(4), 8)
    assert_equal(adapter(2), 7)
    assert_equal(adapter(0), 13)
