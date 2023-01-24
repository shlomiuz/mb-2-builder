#!/usr/bin/env python
import datetime
import logging
import os
import random as rnd
import shutil
import uuid
import json
import numpy as np
import pandas as pd
from docopt import docopt
from pathlib2 import Path

from spotad import parameters, context
from spotad.models import factories as model_factories
from spotad.models.training import GridSearchingTrainer
from spotad.clients.s3 import S3

__doc__ = """
    Train Model

    Usage:
       ./train-model.py [options]

       Options:
         -w, --working-dir PATH      [default: /mnt/spotad/inprocess] working directory, base dir for everything else
         -c, --data-dir PATH         [default: data] data containing header, train, test
         -f, --files-dir PATH        [default: files] additional files folder relative to the working dir 
         -o, --output-dir PATH       [default: Results] results folder, relative to the working dir
         
         --seed X                    [default: 1] random seed ( 0 doesn't work with ranger)
         -t, --threads X             [default: 1] use multiple threads to train models in parallel
         -l, --log-level LEVEL       [default: INFO] logging level
         -m, --override-model NAME   name a model to train, instead of reading from params file
         --log-file FILE_NAME        log file name (instead of stdout)

"""

# read command line arguments
args = docopt(__doc__)

random_seed = int(args['--seed'])
# fixate random seed
rnd.seed(random_seed)
np.random.seed(random_seed)

# init logging
FORMAT = '%(asctime)-15s %(levelname)-7s [%(name)s] %(message)s'
if args['--log-file'] is None:
    logging.basicConfig(format=FORMAT, level=args['--log-level'])
else:
    logging.basicConfig(format=FORMAT, level=args['--log-level'], filename=args['--log-file'])

logger = logging.getLogger('train-model')

# training arguments
threads = int(args['--threads'])

# working folders
base_folder = Path(args['--working-dir'])
output_folder = base_folder / args['--output-dir']
data_folder = base_folder / args['--data-dir']
files_folder = base_folder / args['--files-dir']

s3_client = S3(logger)
s3_base_path = os.environ['S3_BASE_PATH']
s3_client.download_files_in_shell(f'{s3_base_path}/train/', f'{data_folder}/train/')
s3_client.download_files_in_shell(f'{s3_base_path}/test/', f'{data_folder}/test/')
s3_client.download_files_in_shell(f'{s3_base_path}/header/',  f'{data_folder}/header/')
s3_client.download_files_in_shell(f'{s3_base_path}/params/', f'{files_folder}/')

# Make sure the output directory exists
if not output_folder.exists():
    os.makedirs(output_folder.as_posix())

# resolve what goals should be trained
params_file = next(files_folder.glob('*[Pp]arameters*')).as_posix()

with context.log("load Params file: %s", params_file, logger=logger):
    params = parameters.from_file(params_file)

requested_goals = list(map(lambda g: g.lower(), params['Goal']))
exchanges = params['Exchange']

# read algorithm name from params, unless specified otherwise
model_name = params['Algorithm'][0]  # Assume only one algorithm is specified
if args['--override-model'] is not None:
    model_name = args['--override-model']

logger.info("Selected Algorithm: {}".format(model_name))

# A specific algorithm output folder
algo_folder = output_folder / model_name.upper()
tmp_folder = algo_folder / 'tmp'

# if the algorithm directory already exists, delete it
if algo_folder.exists():
    shutil.rmtree(algo_folder.as_posix())
os.makedirs(algo_folder.as_posix())

# For XGBF change the model name to XGB
model_name_org = model_name
model_name = 'XGB' if model_name == 'XGBF' else model_name


# based on model name, get the appropriate factory
model_factory = model_factories.from_name(model_name, tmp_folder=tmp_folder, algo_folder=algo_folder)
multi_label_data_group = model_factory.load_data_group(data_folder)

# validate enough data exists per goal
valid_goals = multi_label_data_group.filter_out_invalid_labels(requested_goals)

if len(valid_goals) == 0:
    logger.warn("No valid goals found to train, exiting")
    exit(0)


def train_goal(model_factory, data_group, goal, work_folder, model_name_org):
    # create inner folder. specific for this goal
    goal_folder = work_folder / goal
    os.makedirs(goal_folder.as_posix())

    model_id = str(uuid.uuid4()).replace('-', '')
    timestamp = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    output_file_postfix = '_' + model_id + '_' + timestamp

    grid_search = GridSearchingTrainer(data_group)
    grid_results = grid_search(model_factory, num_threads=threads)

    grid_search_summary_file = (goal_folder / ('gs_results_{}.csv'.format(model_id))).as_posix()
    logger.info("Saving grid search information to: {}".format(grid_search_summary_file))
    grid_results.save_as_csv(grid_search_summary_file)

    if grid_results.any_valid():
        best_config = grid_results.best_config()
        best_scores = grid_results.scores_for_config(best_config)

        logger.info("Best model found for goal: {0} with configuration: {1}".format(goal, best_config))

        model = model_factory.fresh(best_config)

        # train the final model on the full data
        with data_group.unified_set() as full_data_set:
            model.fit(full_data_set, is_final=True)

        model_file_name = 'model{}'.format(output_file_postfix)
        model_path = (goal_folder / model_file_name).as_posix()
        logger.info("Saving model to to: {}".format(model_path))
        model.save(model_path, confusion={'true_positive': best_scores['true_positive_test'],
                                          'false_negative': best_scores['false_negative_test']})

        keymap_file = (goal_folder / 'keymap{}'.format(output_file_postfix)).as_posix()
        model.save_keymap(keymap_file)

        # Generate refresh descriptor if this model type supports it
        if model_factory.supports_refresh():
            refresh_descriptor_path = (goal_folder / (model_id + '_refresh_desc.json')).as_posix()
            with open(refresh_descriptor_path, 'w') as f:
                logger.info("Creating refresh descriptor: {}".format(refresh_descriptor_path))
                json.dump({'campaign_ids': list(map(int, params['Campaign'])),
                           'goal': goal,
                           'exchanges': exchanges,
                           'model_id': model_id,
                           'model_type': model_factory.model_name,
                           'model_file_name': model_file_name,
                           'extra': model_factory.extra_refresh_metadata(model),
                           'time_frames': [{
                               'from': datetime.datetime.strptime(params['From_Date'], "%Y%m%d").isoformat(),
                               'to': datetime.datetime.strptime(params['To_Date'], "%Y%m%d").isoformat()
                           }]
                           }, f)

        del model
        return [model_id, model_name_org, goal,
                best_scores['roc_auc_test'],
                best_scores['log_loss_test'],
                best_scores['f1_test'],
                'success',
                'model;keymap']
        # return [model_id, model_factory.model_name, goal,
        #         best_scores['roc_auc_test'],
        #         best_scores['log_loss_test'],
        #         best_scores['f1_test'],
        #         'success',
        #         'model;keymap']
    else:
        logger.info("No valid model found, for goal: {}".format(goal))
        return [model_id, model_name_org, goal, '', '', '', 'invalid', '']
        # return [model_id, model_factory.model_name, goal, '', '', '', 'invalid', '']


full_stats = pd.DataFrame(columns=['Model_ID', 'Algorithm', 'Goal', 'AUC', 'LogLoss', 'Accuracy', 'Status', 'Files'])
for goal in valid_goals:
    with multi_label_data_group.focus_on_set(goal) as data_group:
        goal_stats = train_goal(model_factory, data_group, goal, algo_folder, model_name_org)
        full_stats.loc[len(full_stats)] = goal_stats

# write final parameters
if len(full_stats) > 0:
    parameters_file = (algo_folder / 'Parameters.csv').as_posix()
    logger.info("Writing parameters file: %s", parameters_file)
    pd.DataFrame(data=full_stats).to_csv(parameters_file, index=False)
else:
    logger.warn("No models were created, parameters file will not be written")

# Delete the final remains of the temp data
if tmp_folder.exists():
    shutil.rmtree(tmp_folder.as_posix())
logger.info("Process finished")
exit(0)
