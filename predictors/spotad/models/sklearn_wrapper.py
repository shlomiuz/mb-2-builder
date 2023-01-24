from copy import deepcopy
import numpy as np
from spotad.models.common import AbstractModel


class SKLearnModel(AbstractModel):
    def __init__(self, classifier_factory, config):
        super(SKLearnModel, self).__init__()
        self._config = config
        self._classifier = classifier_factory(**config)
        self._header = None

    def fit(self, data_set, is_final=False):
        self._classifier.fit(data_set.X, data_set.y)
        self._header = data_set.header  # save the header it was trained for

    def evaluate(self, data_group):

        scores = dict()

        for name in data_group.names:
            ds = data_group[name]

            try:
                predict_prob = self._classifier.predict_proba(ds.X)[:, 1]
            except:
                predict_prob = None

            predict = self._classifier.predict(ds.X)

            scores.update(super(SKLearnModel,self).compute_data_set_scores(name, ds.y, predict, predict_prob))

        return scores

    def __del__(self):
        self._classifier = None
        self._header = None

    def save(self, path, **kwargs):
        model_dict = deepcopy(kwargs)
        model_dict['model'] = self._classifier
        model_dict['keymap'] = {c: i for i, c in enumerate(self._header)}
        np.save(path, model_dict)

    def save_keymap(self, path):
        # WORKAROUND - RTB expects this file, even when it's not needed
        np.save(path, np.array([np.nan]))

