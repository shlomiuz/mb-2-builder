import numpy as np
from collections import namedtuple
import os
import uuid
import shlex
import logging
from spotad import context
import subprocess
from pathlib2 import Path
import shutil
from spotad.models.common import AbstractModel
from spotad.predict.NB import BayesModel
from itertools import groupby
from operator import itemgetter
from spotad.data.poly2bayes import Poly2BayesData

logger = logging.getLogger("Poly2BayesModel")

_TRAIN_COMMAND_TEMPLATE = """
{spark_submit_path} \
 --class co.spotad.ds.video.TrainModel \
 --conf spark.driver.memory=32g \
 --conf spark.executor.memory=8g \
 --conf spark.executor.cores=16 \
 --conf spark.executor.instance=16 \
 --conf spark.hadoop.fs.s3n.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
 --conf spark.hadoop.fs.s3a.canned.acl=BucketOwnerFullControl \
 --conf spark.hadoop.fs.s3a.acl.default=BucketOwnerFullControl \
 {train_jar_path} \
 --header {header_path} \
 --data {data_path} \
 --output {output_path} \
 --goal {goal} \
 --rarity-threshold {rarity_threshold}
 --threads 1 \
 --master "local[*]" \
 --compute-cr-conversion {compute_cr_conversion}
"""

_TRAIN_COMMAND_TEMPLATE_NO_CR_ADAPTER = """
{spark_submit_path} \
 --class co.spotad.ds.video.TrainModel \
 --conf spark.driver.memory=32g \
 --conf spark.executor.memory=8g \
 --conf spark.executor.cores=16 \
 --conf spark.executor.instance=16 \
 --conf spark.hadoop.fs.s3n.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
 --conf spark.hadoop.fs.s3a.canned.acl=BucketOwnerFullControl \
 --conf spark.hadoop.fs.s3a.acl.default=BucketOwnerFullControl \
 {train_jar_path} \
 --header {header_path} \
 --data {data_path} \
 --output {output_path} \
 --goal {goal} \
 --rarity-threshold {rarity_threshold}
 --threads 1 \
 --master "local[*]" \
 --dummy {compute_cr_conversion}
"""

_TRAIN_COMMAND_TEMPLATE_NO_COMBINATIONS_NO_CR_ADAPTER = """
{spark_submit_path} \
 --class co.spotad.ds.video.TrainModel \
 --conf spark.driver.memory=32g \
 --conf spark.executor.memory=8g \
 --conf spark.executor.cores=16 \
 --conf spark.executor.instance=16 \
 --conf spark.hadoop.fs.s3n.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
 --conf spark.hadoop.fs.s3a.canned.acl=BucketOwnerFullControl \
 --conf spark.hadoop.fs.s3a.acl.default=BucketOwnerFullControl \
 {train_jar_path} \
 --header {header_path} \
 --data {data_path} \
 --output {output_path} \
 --goal {goal} \
 --rarity-threshold {rarity_threshold}
 --threads 1 \
 --master "local[*]" \
 --dummy {compute_cr_conversion} \
 --no-combinations true
"""

_TRAIN_COMMAND_TEMPLATE_NO_COMBINATIONS = """
{spark_submit_path} \
 --class co.spotad.ds.video.TrainModel \
 --conf spark.driver.memory=32g \
 --conf spark.executor.memory=8g \
 --conf spark.executor.cores=16 \
 --conf spark.executor.instance=16 \
 --conf spark.hadoop.fs.s3n.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
 --conf spark.hadoop.fs.s3a.canned.acl=BucketOwnerFullControl \
 --conf spark.hadoop.fs.s3a.acl.default=BucketOwnerFullControl \
 {train_jar_path} \
 --header {header_path} \
 --data {data_path} \
 --output {output_path} \
 --goal {goal} \
 --rarity-threshold {rarity_threshold}
 --threads 1 \
 --master "local[*]" \
 --compute-cr-conversion {compute_cr_conversion}
 --no-combinations true
"""

_FNULL = open(os.devnull, 'w')  # /dev/null stream

PredictionAndProbability = namedtuple("PredictionAndProbability", ['predictions', 'probabilities'])


class Poly2BayesModel(AbstractModel):
    def __init__(self, spark_home_path, rarity_threshold = 1,tmp_folder_path=Path('/tmp')):
        """
        
        :param pathlib2.Path spark_home_path: 
        :param pathlib2.Path tmp_folder_path: 
        """
        super(Poly2BayesModel, self).__init__()
        self._tmp_folder_path = tmp_folder_path
        self._rarity_threshold = rarity_threshold
        self._spark_submit_path = spark_home_path / 'bin' / 'spark-submit'
        self._train_jar_path = Path(os.path.join(os.path.dirname(__file__), '_c/ds-tiny-etl-1.0.23-fat.jar'))
        self._model_file = None
        self._predictor = None

    def fit(self, data_set, is_final=False):
        """
        
        :param Poly2BayesData data_set: 
        :return: 
        """
        assert type(data_set) is Poly2BayesData

        # execute external train and generate model file
        model_id = uuid.uuid4()

        model_file_path = self._tmp_folder_path / (str(model_id) + '.zip')
        model_working_dir = self._tmp_folder_path / (str(model_id) + '_wd')

        # Prepare a tmp folder if it doesnt exists
        if not self._tmp_folder_path.exists():
            logger.debug('Creating the temporary directory that did not yet exist: %s',
                         self._tmp_folder_path.as_posix())
            os.makedirs(self._tmp_folder_path.as_posix())

        logger.debug('Creating a temporary working directory for model training: %s', model_working_dir.as_posix())
        os.makedirs(model_working_dir.as_posix())

        train_params = {
            'spark_submit_path': self._spark_submit_path.as_posix(),
            'train_jar_path': self._train_jar_path.as_posix(),
            'header_path': data_set.header_path,
            'rarity_threshold': self._rarity_threshold,
            'data_path': data_set.data_path,
            'output_path': model_file_path.as_posix(),
            'goal': data_set.goal,
            'compute_cr_conversion': is_final
        }

        train_command = shlex.split(_TRAIN_COMMAND_TEMPLATE_NO_COMBINATIONS_NO_CR_ADAPTER.format(**train_params))
        logger.info('executing: %s', ' '.join(train_command))

        with context.log('Training a model', logger=logger):

            # in case debug logging is enabled, don't forward the ranger stdout to /dev/null
            external_command_out_stream = _FNULL
            if logger.isEnabledFor(logging.DEBUG):
                external_command_out_stream = None  # will cause output to go to this process stdout

            train_process = subprocess.Popen(
                train_command,
                stdout=external_command_out_stream,  # forward process stdout to /dev/null
                cwd=model_working_dir.as_posix())  # working directory
            status = train_process.wait()

            if status != 0:
                logger.error("Training failed")
                shutil.rmtree(model_working_dir.as_posix())
                return

        logger.debug('Removing the temporary working directory for model training: %s', model_working_dir.as_posix())
        shutil.rmtree(model_working_dir.as_posix())

        self._model_file = model_file_path
        logger.debug('Created a temporary model file: %s', self._model_file.as_posix())

        # create a predictor based on the generator model file
        self._predictor = BayesModel(self._model_file.as_posix(), 'historical')

    def predict(self, data_set):
        """
        
        :param Poly2BayesData data_set: 
        :return: 
        """
        data = data_set.get_focused_data()

        X = data.X
        num_rows = X.shape[0]

        # convert the entire X matrix to a dictionary, using getrow inside the loop is very expensive
        sparse_X = {k: list(map(itemgetter(1), g)) for k, g in groupby(zip(*X.nonzero()), itemgetter(0))}
        prob_and_pred = list()

        for row_index in range(num_rows):
            example_indices = sparse_X.get(row_index, {})
            prob, pred = self._predictor.predict_from_indices(example_indices)
            prob_and_pred.append((prob, pred))

        prediction_probability, prediction = zip(*prob_and_pred)

        prediction_probability = np.asarray(prediction_probability, dtype=np.float64).reshape(
            (len(prediction_probability), 1))
        prediction = np.asarray(prediction, dtype=np.uint8).reshape((len(prediction), 1))

        return PredictionAndProbability(prediction, prediction_probability)

    def evaluate(self, data_group):

        scores = dict()

        for name in data_group.names:
            ds = data_group[name]

            with context.log('Evaluating model on %s', name, logger=logger):
                pnp = self.predict(ds)
                scores.update(super(Poly2BayesModel, self).compute_data_set_scores(name,
                                                                                   ds.y,
                                                                                   pnp.predictions,
                                                                                   pnp.probabilities))

        return scores

    def save(self, path, **kwargs):
        logger.info('Saving model to: %s.zip', path)
        with context.log("copy model from temporary folder to: %s.zip", path, logger=logger):
            shutil.copy(self._model_file.as_posix(), path + '.zip')

    def __del__(self):
        if self._model_file is not None:
            logger.debug('Removing the temporary model file: %s', self._model_file.as_posix())
            os.remove(self._model_file.as_posix())
            self._model_file = None
            self._predictor = None

        # if tmp folder is empty, delete it (if another model will need it, it will create it)
        if os.listdir(self._tmp_folder_path.as_posix()) == []:
            logger.debug('Temporary directory is now empty, removing it as well: %s', self._tmp_folder_path.as_posix())
            shutil.rmtree(self._tmp_folder_path.as_posix())

    def save_keymap(self, path):
        # WORKAROUND - RTB expects this file, even when it's not needed
        logger.debug('Creating a dummy keymap file at: %s', path)
        with open(path, 'w') as f:
            f.write("I'm a workaround, all the information is stored in the model file")
