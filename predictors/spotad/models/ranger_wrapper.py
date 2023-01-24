import numpy as np
from collections import namedtuple
import os
import re
import uuid
import shlex
import logging
from spotad import context
import subprocess
from pathlib2 import Path
from operator import methodcaller
import shutil
from spotad.models.common import AbstractModel

_TRAIN_COMMAND_TEMPLATE = "{ranger_bin} --file {train_file} --treetype 1 --probability" + \
                          " --memmode 1 --seed {random_seed} --caseweights {case_weights_file}" + \
                          " --depvarname {label_name} --write --outprefix {out_name} --splitrule 1" + \
                          " --ntree {ntree} --fraction {fraction} --targetpartitionsize {min_node_size}"

_PREDICT_COMMAND_TEMPLATE = "{ranger_bin} --file {prediction_file} --seed {random_seed}" + \
                            " --predict {out-name}.forest --outprefix {out-name}"

_FNULL = open(os.devnull, 'w')  # /dev/null stream

WHITESPACE = re.compile(r'\s+')

PredictionAndProbability = namedtuple("PredictionAndProbability", ['predictions', 'probabilities'])

logger = logging.getLogger("RangerModel")


def _load_prediction(file_name):
    """
    Load the predictions and probabilities of those predictions from the ranger (non standard) format 
    
    :param str file_name: path to ranger generated prediction file
    :return: prediction information with predictions and probabilities
    :rtype: PredictionAndProbability
    """
    with open(file_name, 'r') as f:
        next(f)  # ignore the explanation line
        header_line = next(f).strip()
        # figure out what index represents the positive labels
        try:
            true_label_index = WHITESPACE.split(header_line).index("1")
        except ValueError:  # everything is zero so no column named 1 exists
            true_label_index = None

        next(f)  # ignore empty line

        pred_and_prob = list(map(lambda p: (p, 1 if p > 0.5 else 0),
                            [float(WHITESPACE.split(line)[true_label_index])
                             if true_label_index is not None else 0.0  # assign all probs to 0 if no column for 1 exists
                             for line in map(methodcaller('strip'), f)]))

        prediction_probability, prediction = zip(*pred_and_prob)

        prediction_probability = np.asarray(prediction_probability, dtype=np.float64).reshape(
            (len(prediction_probability), 1))
        prediction = np.asarray(prediction, dtype=np.uint8).reshape((len(prediction), 1))

        return PredictionAndProbability(prediction, prediction_probability)


class RangerModel(AbstractModel):
    def __init__(self, config, tmp_folder=Path('/tmp'), ranger_bin=None, random_seed=1):
        super(RangerModel, self).__init__()
        self._config = config
        self._tmp_folder = tmp_folder
        self._random_seed = random_seed
        self._ranger_bin = ranger_bin
        self._header = None
        self._model_file = None

    def _get_ranger_bin(self):
        """
        Get a ranger binary path, if no manual path was specified, return the default
        :return: the path
        :rtype: str
        """
        if self._ranger_bin is None:
            return os.path.join(os.path.dirname(__file__), '_c/gzranger')
        else:
            return self._ranger_bin

    def fit(self, data_set, is_final=False):
        self._header = data_set.header
        # execute train and generate model file
        self._model_name = uuid.uuid4()

        # make sure the ranger bin has an executable permission
        os.chmod(self._get_ranger_bin(), 0o777)

        ranger_train_params = {
            'ranger_bin': self._get_ranger_bin(),
            'train_file': data_set.data_file,
            'case_weights_file': data_set.case_weights_file,
            'label_name': data_set.label_name,
            'random_seed': self._random_seed,
            'fraction': self._config['fraction'],
            'ntree': self._config['ntree'],
            'min_node_size': self._config['min_target_node_size'],
            'out_name': self._model_name
        }

        ranger_command = shlex.split(_TRAIN_COMMAND_TEMPLATE.format(**ranger_train_params))
        logger.debug('executing: %s', ' '.join(ranger_command))

        with context.log("Training a ranger model for ntree:%d, fraction:%f min_target_node_size:%d from: %s",
                         self._config['ntree'], self._config['fraction'], self._config['min_target_node_size'],
                         data_set.data_file, logger=logger):

            # in case debug logging is enabled, don't forward the ranger stdout to /dev/null
            ranger_out_stream = _FNULL
            if logger.isEnabledFor(logging.DEBUG):
                ranger_out_stream = None

            ranger_process = subprocess.Popen(
                ranger_command,
                stdout=ranger_out_stream,  # forward process stdout to /dev/null
                cwd=self._tmp_folder.as_posix())  # working directory
            status = ranger_process.wait()
            if status != 0:
                logger.error("Training failed for ntree:%d, fraction:%f min_target_node_size:%d from: %s",
                             self._config['ntree'], self._config['fraction'], self._config['min_target_node_size'],
                             data_set.data_file)
                return None

        self._model_file = self._tmp_folder / '{}.forest'.format(self._model_name)
        logger.info('Created a temporary model file: %s', self._model_file.as_posix())

    def predict(self, data_set):
        """
        Run a prediction of the model of some data set.
        Will raise an error if used outside a 'with' block
        
        :param spotad.data.ranger_wrapper.RangerDataSet data_set: a path to a data set
        :return: predictions and the corresponding probabilities of running the model on the data set
        :rtype: PredictionAndProbability
        """
        if self._model_file is None: raise AssertionError("Prediction can only be done with a trained model")

        ranger_predict_params = {
            'ranger_bin': self._get_ranger_bin(),
            'random_seed': self._random_seed,
            'prediction_file': data_set.data_file,
            'out-name': self._model_name
        }

        ranger_command = shlex.split(_PREDICT_COMMAND_TEMPLATE.format(**ranger_predict_params))
        logger.debug('executing: %s', ' '.join(ranger_command))

        ranger_process = subprocess.Popen(
            ranger_command,
            stdout=_FNULL,  # forward process stdout to /dev/null
            cwd=self._tmp_folder.as_posix())  # working directory
        status = ranger_process.wait()
        if status != 0:
            raise Exception('Prediction process failed')

        prediction_file = (self._tmp_folder / "{}.prediction".format(self._model_name)).as_posix()
        predictions_and_probabilities = _load_prediction(prediction_file)

        # delete the prediction file that is no longer needed
        os.remove(prediction_file)

        return predictions_and_probabilities

    def evaluate(self, data_group):

        scores = dict()

        for name in data_group.names:
            ds = data_group[name]

            with context.log('Evaluating model on %s', name, logger=logger):
                pnp = self.predict(ds)
                scores.update(super(RangerModel,self).compute_data_set_scores(name,
                                                                              ds.y,
                                                                              pnp.predictions,
                                                                              pnp.probabilities))

        return scores

    def __del__(self):
        if self._model_file is not None:
            os.remove(self._model_file.as_posix())
            self._model_file = None

    def save(self, path, **kwargs):
        logger.info('Saving model to: %s', path)
        with context.log("copy model from temporary folder to: %s", path, logger=logger):
            shutil.copy(self._model_file.as_posix(), path)

    def save_keymap(self, path):
        # save the keymap corresponding to the model
        with open(path, 'w') as out, \
                context.log("writing keymap file: %s", path, logger=logger):
            out.write(','.join(self._header))
            out.write('\n')
            out.write(','.join(map(str, range(1, len(self._header) + 1))))
