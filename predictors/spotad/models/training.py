from itertools import product
from spotad import context
import logging
import pandas as pd
import numpy as np

eval_logger = logging.getLogger("TrainEvalDriver")


class SingleConfigurationTrainer(object):
    def __init__(self, data_group, train_set_name='train'):
        self._data_group = data_group
        self._train_set_name = train_set_name

    def __call__(self, model_factory, config):
        model = model_factory.fresh(config)

        # train using train set
        with context.log("Training model configuration: {}".format(config), logger=eval_logger):
            model.fit(self._data_group[self._train_set_name])

        with context.log("Evaluating model configuration: {}".format(config), logger=eval_logger):
            evaluation = model.evaluate(self._data_group)

        # delete model, hopefully this might free up some memory
        del model

        return evaluation

def flatten_config(config_space):
    names, options = zip(*config_space.iteritems())  # unzip
    return [dict(zip(names, op)) for op in product(*options)]


class GridSearchingTrainer(object):
    def __init__(self, data_group, train_set_name='train'):
        self._single_trainer = SingleConfigurationTrainer(data_group, train_set_name)

    def __call__(self, model_factory, num_threads=1):
        # convert config space into individual configurations
        hyper_parameter_space = model_factory.hyper_parameter_space
        if len(hyper_parameter_space) > 0:
            configs = flatten_config(hyper_parameter_space)
        else:
            configs = [{}] # if hyper parameter space is empty, there is a single empty configuration

        results = []
        if num_threads > 1:  # use a thread pool to train in parallel
            from concurrent.futures import ThreadPoolExecutor, as_completed

            pool = ThreadPoolExecutor(num_threads)
            futures = [pool.submit(self._single_trainer, model_factory, config) for config in configs]
            results.extend(zip(configs, [r.result() for r in as_completed(futures)]))
            pool.shutdown(wait=False)  # nothing to wait for, already waited for all jobs

        else:  # use a simple loop and train each configuration one after the other
            for config in configs:
                result = self._single_trainer(model_factory, config)
                results.append((config, result))

        return GridSearchResults(results)


class GridSearchResults(object):
    def __init__(self, config_metrics_pairs):
        c, m = zip(*config_metrics_pairs)
        self._data = pd.DataFrame(list(c)).join(pd.DataFrame(list(m)))
        self._best_config = None

        # compute validity
        self._data['roc_auc_diff'] = abs(1- self._data['roc_auc_test']/(self._data['roc_auc_train']+0.00000000001))
        self._data['valid_roc_diff'] = self._data['roc_auc_diff'] <= 0.4 # 0.1
        self._data['valid_roc_score'] = self._data['roc_auc_test'] >= 0.5 #0.6
        self._data['valid_model'] = np.logical_and(self._data['valid_roc_diff'], self._data['valid_roc_score'])

        self._data['best'] = False

        if self._data['valid_model'].sum() > 0:
            # choose best model
            best_model_row_index = self._data[self._data['valid_model']]['roc_auc_test'].idxmax()
            self._best_config = config_metrics_pairs[best_model_row_index][0]

            # write results with the indication of what model was chosen as the best
            self._data.set_value(best_model_row_index, 'best', True) # indicate that this is the best model

    def scores_for_config(self, config):
        return self._data.loc[(self._data[list(config)] == pd.Series(config)).all(axis=1)].to_dict(orient='records')[0]

    def any_valid(self):
        return self._data['valid_model'].sum() > 0

    def best_config(self):
       return self._best_config

    def save_as_csv(self, file_name):
      self._data.to_csv(file_name, index=False)
