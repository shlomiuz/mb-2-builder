from nose.tools import *
from spotad.ranger.algo import _load_prediction as load_prediction
from os import path

def test_load_prediction_variant1():
    pred_and_prob = load_prediction(path.join(path.dirname(__file__),'dummy_1.prediction'))
    p = pred_and_prob.probabilities

    assert_equals(p.shape, (5,1))

    assert_equals(p[0][0], 0.0)
    assert_equals(p[1][0], 1.0)
    assert_equals(p[2][0], 0.025)
    assert_almost_equal(p[3][0], 7.51 / 100000.0, places=5)
    assert_almost_equal(p[4][0], 1.0 / 3.0, places=5)

    pred = pred_and_prob.predictions

    assert_equals(pred[0][0], 0)
    assert_equals(pred[1][0], 1)
    assert_equals(pred[2][0], 0)
    assert_equals(pred[3][0], 0)
    assert_equals(pred[4][0], 0)


def test_load_prediction_variant2():
    pred_and_prob = load_prediction(path.join(path.dirname(__file__),'dummy_2.prediction'))
    p = pred_and_prob.probabilities

    assert_equals(p.shape, (5,1))

    assert_equals(p[0][0], 0.0)
    assert_equals(p[1][0], 1.0)
    assert_equals(p[2][0], 0.025)
    assert_almost_equal(p[3][0], 7.51 / 100000.0, places=5)
    assert_almost_equal(p[4][0], 1.0 / 3.0, places=5)

    pred = pred_and_prob.predictions

    assert_equals(pred[0][0], 0)
    assert_equals(pred[1][0], 1)
    assert_equals(pred[2][0], 0)
    assert_equals(pred[3][0], 0)
    assert_equals(pred[4][0], 0)

