from multiprocessing import cpu_count
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from spotad.data import io as spio
from spotad.data.scipy_wrapper import SciPyMultiLabelDataGroup
from spotad.data.poly2bayes import Poly2BayesData
from spotad.data.ranger_wrapper import RangerMultiLabelDataGroup
from spotad.data.xgboost_wrapper import XGBoostMultiLabelDataGroup
from spotad.models.sklearn_wrapper import SKLearnModel
from spotad.models.ranger_wrapper import RangerModel
from spotad.models.xgboost_wrapper import XGBoostModel
from spotad.models.poly2bayes import Poly2BayesModel
from spotad.models.common import AbstractModelFactory
from pathlib2 import Path
from functools import partial
from copy import deepcopy
import os


class LinearFactory(AbstractModelFactory):
    def __init__(self, algo_folder, tmp_folder, name, fit_intercept=False, n_jobs=cpu_count()):
        super(LinearFactory, self).__init__()  # call super ctor
        self._fit_intercept = fit_intercept
        self._n_jobs = n_jobs
        self._name = name

    def load_data_group(self, data_folder):
        header_path = data_folder / 'header'
        train_path = data_folder / 'train'
        test_path = data_folder / 'test'

        # load train and test sets
        sparse_train = spio.load_sparse_data_set(header_path, train_path)
        sparse_test = spio.load_sparse_data_set(header_path, test_path)

        return SciPyMultiLabelDataGroup({'train': sparse_train, 'test': sparse_test})

    @property
    def model_name(self):
        return self._name

    def fresh(self, config):
        """
        :param dict config: 
        :return: a fresh model
        :rtype: 
        """
        # add specific properties
        full_config = dict(list(config.items()) + [('fit_intercept', self._fit_intercept), ('n_jobs', self._n_jobs)])
        return SKLearnModel(SGDClassifier, full_config)


#
# Note: This factory is no longer used, it was replaced by Poly2BayesImpersonatingNB
#
class NaiveBayesFactory(AbstractModelFactory):
    def __init__(self, algo_folder, tmp_folder, binarize=None):
        super(NaiveBayesFactory, self).__init__()  # call super ctor
        self._binarize = binarize

    def load_data_group(self, data_folder):
        header_path = data_folder / 'header'
        train_path = data_folder / 'train'
        test_path = data_folder / 'test'

        # load train and test sets
        sparse_train = spio.load_sparse_data_set(header_path, train_path)
        sparse_test = spio.load_sparse_data_set(header_path, test_path)

        return SciPyMultiLabelDataGroup({'train': sparse_train, 'test': sparse_test})

    @property
    def model_name(self):
        return "NB"

    def fresh(self, config):
        """
        :param dict config: 
        :return: a fresh model
        :rtype: 
        """
        full_config = dict(list(config.items()) + [('binarize', self._binarize)])
        return SKLearnModel(BernoulliNB, full_config)


class RandomForestFactory(AbstractModelFactory):
    def __init__(self, algo_folder, tmp_folder):
        super(RandomForestFactory, self).__init__()  # call super ctor
        self._tmp_folder = tmp_folder

    def load_data_group(self, data_folder):
        header_path = data_folder / 'header'
        train_path = data_folder / 'train'
        test_path = data_folder / 'test'

        # load train and test sets
        sparse_train = spio.load_sparse_data_set(header_path, train_path)
        sparse_test = spio.load_sparse_data_set(header_path, test_path)

        return RangerMultiLabelDataGroup({'train': sparse_train, 'test': sparse_test}, self._tmp_folder)

    @property
    def model_name(self):
        return "RF"

    def fresh(self, config):
        """
        :param dict config: 
        :return: a fresh model
        :rtype: 
        """
        return RangerModel(config, self._tmp_folder)


class XGBoostFactory(AbstractModelFactory):
    def __init__(self,
                 algo_folder,
                 tmp_folder='',
                 booster='gbtree',
                 subsample=0.8,
                 colsample_bytree=0.8,
                 objective='binary:logistic',
                 eval_metric='auc',
                 num_boost_rounds=100):
        super(XGBoostFactory, self).__init__()
        self._algo_folder = algo_folder
        self._static_config = {'booster': booster,
                               'subsample': subsample,
                               'colsample_bytree': colsample_bytree,
                               'objective': objective,
                               'eval_metric': eval_metric,
                               'silent': 1
                               }

        self._num_boost_rounds = num_boost_rounds

    def load_data_group(self, data_folder):
        header_path = data_folder / 'header'
        train_path = data_folder / 'train'
        test_path = data_folder / 'test'

        # load train and test sets
        sparse_train = spio.load_sparse_data_set(header_path, train_path)
        sparse_test = spio.load_sparse_data_set(header_path, test_path)

        return XGBoostMultiLabelDataGroup({'train': sparse_train, 'test': sparse_test})

    @property
    def model_name(self):
        return "XGB"

    def fresh(self, config):
        """
        :param dict config: 
        :return: a fresh model
        :rtype: 
        """
        full_config = deepcopy(self._static_config)
        full_config.update(config)
        return XGBoostModel(full_config, algo_folder=self._algo_folder, num_boost_rounds=self._num_boost_rounds)


class Poly2BayesFactory(AbstractModelFactory):
    def __init__(self, algo_folder, tmp_folder):
        super(Poly2BayesFactory, self).__init__()  # call super ctor
        self._tmp_folder = tmp_folder

    def load_data_group(self, data_folder):
        header_path = data_folder / 'header'
        train_path = data_folder / 'train'
        test_path = data_folder / 'test'

        # load train and test sets
        # lil format is used because it has a getrowview which will allow getting non zero indexes without needing to
        # perform copy of rows (which happens when using getrow to access a specific row)
        sparse_train = spio.load_sparse_data_set(header_path, train_path, sparse_output_format='lil')
        sparse_test = spio.load_sparse_data_set(header_path, test_path, sparse_output_format='lil')

        return Poly2BayesData.load(header_path, train_path, sparse_train, test_path, sparse_test, self._tmp_folder)

    @property
    def model_name(self):
        return "POLY2BAYES"

    def fresh(self, config):
        """
        :param dict config: 
        :return: a fresh model
        :rtype: 
        """

        return Poly2BayesModel(
            spark_home_path=Path(os.getenv('SPARK_HOME', '/home/ubuntu/spark')),
            rarity_threshold=int(config['rarity_threshold']),
            tmp_folder_path=self._tmp_folder
        )


class Poly2BayesImpersonatingNB(Poly2BayesFactory):
    def __init__(self, algo_folder, tmp_folder):
        super(Poly2BayesImpersonatingNB, self).__init__(tmp_folder)  # call super ctor

    @property
    def model_name(self):
        return "NB"

    def hyper_parameter_file(self):
        return 'POLY2BAYES.json'

    def supports_refresh(self):
        return True

    def extra_refresh_metadata(self, model):
        return {'rarity-threshold': model._rarity_threshold}


_FACTORIES_DICT = {
    'nb': Poly2BayesImpersonatingNB,
    'lasso': partial(LinearFactory, name='LASSO'),
    'sgd': partial(LinearFactory, name='SGD'),
    'rf': RandomForestFactory,
    'xgb': XGBoostFactory,
    'poly2bayes': Poly2BayesFactory,
}


def from_name(name, algo_folder, tmp_folder=Path('/tmp')):
    return _FACTORIES_DICT[name.lower()](algo_folder, tmp_folder)
