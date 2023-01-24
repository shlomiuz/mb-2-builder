from spotad.models.common import AbstractModel
import logging
import xgboost as xgb
from spotad import context
import numpy as np
import os

logger = logging.getLogger("XGBoostModel")


class XGBoostModel(AbstractModel):
    def __init__(self, config, algo_folder, num_boost_rounds=10):
        super(XGBoostModel, self).__init__()
        self._config = config
        self._model = None
        self._num_boost_rounds = num_boost_rounds
        self._algo_folder = algo_folder

    def fit(self, data_set, is_final=False):
        # save header of trained data
        self._header = data_set.header
        evallist = [(data_set.data_matrix, 'train')]
        self._model = xgb.train(params=self._config, dtrain=data_set.data_matrix,
                                num_boost_round=self._num_boost_rounds, evals=evallist)
        if is_final is True:
            print("Final")
            self.save_importance()

    def evaluate(self, data_group):
        scores = dict()

        for name in data_group.names:
            ds = data_group[name]

            with context.log('Evaluating model on %s', name, logger=logger):
                #
                probs = self._model.predict(ds.data_matrix)
                self.save_probs(probs, self._algo_folder / 'probs' / name, 'probs')
                preds = np.around(probs)  # Prediction threshold is 0.5
                self.save_probs(preds, self._algo_folder / 'probs' / name, 'preds')
                self.save_probs(ds.y, self._algo_folder / 'probs' / name, 'y')
                scores.update(super(XGBoostModel, self).compute_data_set_scores(name,
                                                                                ds.y,
                                                                                preds,
                                                                                probs))

        return scores

    def save(self, path, **kwargs):
        self._model.save_model(path)

    def save_keymap(self, path):
        with open(path + '.csv', 'w') as out:
            for i, feature_name in enumerate(self._header):
                out.write('{0}$${1}\n'.format(feature_name, i))

    def save_probs(self, probs, path, name):
        parameters_file = path.as_posix()
        logger.info("Writing file: %s", path)
        if not path.exists():
            os.makedirs(parameters_file)
        with open(parameters_file + '/{}.csv'.format(name), 'w') as out:
            for i in probs:
                out.write('{}\n'.format(i))
        np.save(parameters_file + '/{}'.format(name), np.array(probs))

    def save_importance(self):
        path = self._algo_folder / 'importance'
        parameters_file = path.as_posix()
        if not path.exists():
            os.makedirs(parameters_file)
        for i in ['weight', 'gain', 'cover']:
            importance = self._model.get_score(importance_type=i)
            logger.info("Writing file: %s", 'importance_{}'.format(i))
            with open(parameters_file + '/importance_{}.csv'.format(i), 'w') as out:
                for k, v in importance.items():
                    out.write('{0}$${1}\n'.format(k, v))
            # np.save(parameters_file, np.array(importance))