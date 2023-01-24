from nose.tools import *
from ..training import flatten_config


def test_flatten_config():
    dummy_config = {
        'a': [1, 2],
        'b': ["h"],
        'c': ['A', 'B', 'C']
    }

    individual_configs = flatten_config(dummy_config)
    assert_equals(len(individual_configs), 6)

    assert_equals(len(filter(lambda x: x['a'] == 1, individual_configs)), 3)
    assert_equals(len(filter(lambda x: x['a'] == 2, individual_configs)), 3)
    assert_equals(len(filter(lambda x: x['c'] == 'A', individual_configs)), 2)
    assert_equals(len(filter(lambda x: x['c'] == 'B', individual_configs)), 2)
    assert_equals(len(filter(lambda x: x['c'] == 'C', individual_configs)), 2)
    assert_equals(len(filter(lambda x: x['b'] == 'h', individual_configs)), 6)
