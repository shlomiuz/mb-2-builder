import json
from os import path
from sklearn import metrics
import numpy as np


class AbstractModelFactory(object):

    def __init__(self):
        self._hyper_params = None

    def supports_refresh(self):
        return False

    def extra_refresh_metadata(self, model):
        return dict()

    @property
    def model_name(self):
        raise NotImplementedError

    def fresh(self, config):
        raise NotImplementedError

    def hyper_parameter_file(self):
        return self.model_name + '.json'

    @property
    def hyper_parameter_space(self):
        if self._hyper_params is None:
            # load hyper parameters from an external file
            file_name = path.join('hyper-parameters', self.hyper_parameter_file())
            with open(path.join(path.dirname(__file__), file_name),'r') as f:
                self._hyper_params = json.load(f)
        return self._hyper_params


class AbstractModel(object):

    def compute_data_set_scores(self, name, true_labels, predict_labels, predict_prob):
        scores = dict()
        if predict_prob is not None:
            scores['roc_auc_' + name] = metrics.roc_auc_score(true_labels, predict_prob)
            scores['log_loss_' + name] = metrics.log_loss(true_labels, predict_prob)
        else:
            scores['roc_auc_' + name] = np.nan
            scores['log_loss_' + name] = np.nan

        scores['f1_' + name] = metrics.f1_score(true_labels, predict_labels)
        scores['recall_' + name] = metrics.recall_score(true_labels, predict_labels)
        scores['precision_' + name] = metrics.precision_score(true_labels, predict_labels)

        cm = metrics.confusion_matrix(true_labels, predict_labels)

        if (np.sum(cm[1, 1]) > 0) and (np.sum(cm[0, 0]) > 0):
            scores['true_positive_' + name] = cm[1, 1] / float(np.sum(cm[:, 1]))
            scores['false_negative_' + name] = cm[1, 0] / float(np.sum(cm[:, 0]))
        else:
            scores['true_positive_' + name] = 0.
            scores['false_negative_' + name] = 0.

        return scores
