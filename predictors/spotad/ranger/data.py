from spotad import context, timeutils, transform
from spotad.transform import MatrixWithNames
import numpy as np
import pandas as pd
import logging
from scipy.sparse import hstack
from pathlib2 import Path
import math
from sklearn.model_selection import train_test_split
import os
import uuid
import subprocess

_COMBINE_DATA_SETS_COMMAND_TEMPLATE = "head -n -1 {test_file} >> {train_file}"
_COMBINE_DATA_CASE_WEIGHTS_COMMAND_TEMPLATE = "cat {test_file} >> {train_file}"
_CASE_WEIGHTS_FILE_POSTFIX = '.case_weights'
MINIMUM_DATA_PER_LABEL_VALUE = 10


class DataSetReference(object):
    """
    A representation of a specific data set with a specific label.
    It's implemented as a context manger in order to make it easy to cleanup intermediate files that are generated
    for the sake of ranger
    """

    def __init__(self, features_data, feature_names, label_data, label_name, day_ts, timestamp, tmp_folder):
        """
        
        :param numpy.array features_data: the feature matrix 
        :param list feature_names: the names of features (naming every column in the feature data)
        :param numpy.array label_data: the values of the label corresponding to the features
        :param str label_name: the name of the label used in the data set
        :param pandas.Series day_ts: the days corresponding to every example in the data set
        :param pandas.Series timestamp: the timestamps corresponding to every example in the data set
        :param pathlib2.Path tmp_folder: a temporary folder where data set files can be temporarily stored
        """

        self._features_data = features_data
        self._feature_names = feature_names
        self._label_data = label_data
        self._label_name = label_name
        self._day_ts = day_ts
        self._timestamp = timestamp
        self._tmp_folder = tmp_folder

        self._full_set_prepared = False
        self._context_available = False
        self._logger = logging.getLogger('DataSetReference')

        self._y_train = None
        self._y_test = None
        self._train_file = None
        self._test_file = None

        self._enough_data_available = None

    def _write_named_matrix(self, data, file):
        """
        Write a ranger compatible CSV file from a matrix and a list of column names
        
        :param MatrixWithNames data: values matrix and the names of the columns
        :param str file_name: path of the CSV file to write
        """

        with open(file, 'w') as out:
            # write header manually, numpy header writes header as a comment
            out.write('\t'.join(data.column_names))
            out.write('\n')

            # write the data
            np.savetxt(out, data.matrix.astype(np.uint8).todense(), fmt='%1u', delimiter='\t')

    def _write_case_weights(selfs, data, file):
        with open(file, 'w') as out:
            np.savetxt(out, data.reshape((1,data.shape[0])), fmt='%1.5f', delimiter=' ', newline=' ')

    def prepare_full_set(self):
        """
        Create the full train set, this has to be called explicitly for it to work efficiently
        """
        self._validate_context()
        self._full_set_prepared = True

        # append the test set (without the header to the existing train set)
        combine_set_command = _COMBINE_DATA_SETS_COMMAND_TEMPLATE.format(test_file=self._test_file,
                                                                     train_file=self._train_file)


        combine_case_weights_command = _COMBINE_DATA_CASE_WEIGHTS_COMMAND_TEMPLATE.format(
            test_file=self._test_file + _CASE_WEIGHTS_FILE_POSTFIX,
            train_file=self._train_file + _CASE_WEIGHTS_FILE_POSTFIX)


        with context.log("Appending test set to train, to produce a full data set", logger=self._logger):
            self._logger.debug('executing: %s', combine_set_command)
            status = subprocess.call(combine_set_command, shell=True)  # working directory
            if status != 0:
                raise Exception("Failed creating a full data set")

            self._logger.debug('executing: %s', combine_case_weights_command)
            status = subprocess.call(combine_case_weights_command, shell=True)  # working directory
            if status != 0:
                raise Exception("Failed creating a full data set case weights")

    def _validate_context(self):
        """
        Utility that raises an exception if called while accessed from outside of a 'with' block 
        """
        if not self._context_available:
            raise Exception('can only be used inside a with statement')
        if not self._enough_data_available:
            raise Exception('There is not enough data in the data set')

    @property
    def train_set(self):
        self._validate_context()
        if self._full_set_prepared:
            raise Exception('prepare full set was called so this is no longer available')
        return self._train_file

    @property
    def test_set(self):
        self._validate_context()
        return self._test_file

    @property
    def full_set(self):
        self._validate_context()
        if not self._full_set_prepared:
            raise Exception('prepare full set must be called before this is available')
        return self._train_file

    @property
    def train_gold_labels(self):
        self._validate_context()
        return self._y_train

    @property
    def test_gold_labels(self):
        self._validate_context()
        return self._y_test

    @property
    def feature_names(self):
        self._validate_context()
        return self._feature_names

    def enough_data_available(self):
        if self._enough_data_available is None:
            raise Exception('can only be used inside a with statement')
        return self._enough_data_available


    def _validate_label_diversity(self, y):
         # validate enough data exists
        label_counts = np.bincount(y.reshape(y.shape[0]))
        if len(label_counts) < 2 \
                or label_counts[0] < MINIMUM_DATA_PER_LABEL_VALUE \
                or label_counts[1] < MINIMUM_DATA_PER_LABEL_VALUE:
            return False
        else:
            return True

    def _compute_case_weights(self, y):
        num_samples = y.shape[0]
        return tuple(float(num_samples) / (2 * np.bincount(y.reshape(num_samples))))

    def __enter__(self):
        """
        Split the full data into train and test and write temporary data set files needed by ranger.
        The label column in the files will be named __label__
        
        :return: self
        :rtype: DataSetReference
        """

        y = self._label_data

        # Validate that there is enough label diversity
        self._enough_data_available = self._validate_label_diversity(y)
        if not self._enough_data_available:
            return self # Stop at this point, no need to generate a ranger data set, it won't be useful

        # train test splitting
        if len(self._day_ts.unique()) >= 14:
            dt = np.sort(self._day_ts.unique().astype(np.int64))
            dt_train = dt[range(int(math.floor(0.75 * len(dt))))]
            dt_train_inx = self._day_ts.isin(dt_train.astype(str))
            train_inx = [j for j, s in enumerate(dt_train_inx) if s == True]
            X_train = self._features_data[train_inx, :]
            self._y_train = y[train_inx]
            test_inx = [j for j, s in enumerate(dt_train_inx) if s == False]
            X_test = self._features_data[test_inx, :]
            self._y_test = y[test_inx]
        else:
            X_train, X_test, self._y_train, self._y_test = train_test_split(self._features_data, y,
                                                                            test_size=0.25,
                                                                            random_state=0,
                                                                            stratify=y)


        # Validate that there is enough label diversity even after the split
        self._enough_data_available = self._validate_label_diversity(self._y_train)
        if not self._enough_data_available:
            return self # Stop at this point, no need to generate a ranger data set, it won't be useful

        self._enough_data_available = self._validate_label_diversity(self._y_test)
        if not self._enough_data_available:
            return self # Stop at this point, no need to generate a ranger data set, it won't be useful

        # generate case weights for train and test
        negative_weight, positive_weight = self._compute_case_weights(self._y_train)
        train_case_weights = np.ones(self._y_train.shape[0]) * negative_weight
        train_case_weights[self._y_train.reshape(self._y_train.shape[0]) == 1] = positive_weight

        negative_weight, positive_weight = self._compute_case_weights(self._y_test)
        test_case_weights = np.ones(self._y_test.shape[0]) * negative_weight
        test_case_weights[self._y_test.reshape(self._y_test.shape[0]) == 1] = positive_weight


        train_matrix = MatrixWithNames(
            hstack([X_train, self._y_train], format='lil'),
            self._feature_names + ['__label__']
        )

        test_matrix = MatrixWithNames(
            hstack([X_test, self._y_test], format='lil'),
            self._feature_names + ['__label__']
        )

        # Prepare a tmp folder if it doesnt exists
        if not self._tmp_folder.exists():
            os.makedirs(self._tmp_folder.as_posix())

        # generate unique file names
        unique_id = uuid.uuid4()

        self._train_file = (self._tmp_folder / 'train-{}.csv'.format(unique_id)).as_posix()
        with context.log("Writing train set temporary file: %s", self._train_file):
            self._write_named_matrix(train_matrix, self._train_file)

        train_case_weights_file = self._train_file + _CASE_WEIGHTS_FILE_POSTFIX
        with context.log("Writing train set case weights temporary file: %s", train_case_weights_file):
            self._write_case_weights(train_case_weights, train_case_weights_file)

        self._test_file = (self._tmp_folder / 'test-{}.csv'.format(unique_id)).as_posix()
        with context.log("Writing test set temporary file: %s", self._test_file):
            self._write_named_matrix(test_matrix, self._test_file)

        test_case_weights_file = self._test_file + _CASE_WEIGHTS_FILE_POSTFIX
        with context.log("Writing test set case weights temporary file: %s", test_case_weights_file):
            self._write_case_weights(test_case_weights, test_case_weights_file)

        self._context_available = True
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Cleanup the temporary resources/files created for this dataset
        
        :param exception_type: 
        :param exception_value: 
        :param traceback: 
        """
        if self._y_train is not None:
            del self._y_train
            self._y_train = None

        if self._y_test is not None:
            del self._y_test
            self._y_test = None

        # delete the files created (if any)
        if self._train_file is not None:
            self._logger.info("Deleting train set temporary files: %s, %s",
                              self._train_file,
                              self._train_file + _CASE_WEIGHTS_FILE_POSTFIX)
            os.remove(self._train_file)
            os.remove(self._train_file + _CASE_WEIGHTS_FILE_POSTFIX)
            self._train_file = None

        if self._test_file is not None:
            self._logger.info("Deleting test set temporary files: %s, %s",
                              self._test_file,
                              self._test_file + _CASE_WEIGHTS_FILE_POSTFIX)
            os.remove(self._test_file)
            os.remove(self._test_file + _CASE_WEIGHTS_FILE_POSTFIX)
            self._test_file = None

        self._context_available = False
        self._enough_data_available = None


############################################################################################################
# Columns to be considered for training (the names used are after a feature engineering step is performed) #
############################################################################################################

DATA_LIST_COLUMNS = ['category']

DATA_ATTRIBUTE_COLUMNS = ['impression_position_val', 'impression_viewability', 'app_site_ind', 'domain', 'publisher',
                          'formats', 'device_language', 'device_make', 'device_model',
                          'device_os', 'device_osv', 'loc_country', 'country_state', 'country_state_city', 'zip',
                          'datacenter', 'browser', 'banner', 'ad_type', 'banner_size', 'sub_account_id', 'HOW']


class DataLoader(object):
    """
    Loads data, transforms to a feature ready format and caches the results.  
    """

    def __init__(self, data_file, header_file, labels, value_rarity_threshold=100, tmp_folder=Path('/tmp')):
        """
        
        :param str data_file: the path to csv file containing the data
        :param header_file: the path to a csv file that contains the header that corresponds to the data file
        :param labels: what columns to consider as possible labels and cache
        :param value_rarity_threshold: amount of appearances of values that is considered rare
        :param tmp_folder: a temporary folder where intermediate files can be saved
        """
        self._data_file = data_file
        self._header_file = header_file
        self._labels = labels
        self._logger = logging.getLogger('DataLoader')
        self._value_rarity_threshold = value_rarity_threshold
        self._tmp_folder = tmp_folder

        if len(labels) == 0:
            self._logger.error("No labels specified")
            raise ValueError("no labels provided")
        else:
            self._logger.debug("Requested labels are: %s", ', '.join(labels))

        self._feature_names = None
        self._features_data = None
        self._labels_data = None
        self._day_ts = None
        self._timestamp = None

        self._loaded = False

    # lazy
    def _load(self):
        """
        Loading of data, and its transformation to a feature ready format. The results from this process are cached and further calls to this method are ignored. 
        """
        if self._loaded: return  # we only want this to happen once

        # load the raw training data
        with context.log("load Header file: %s", self._header_file, logger=self._logger):
            headers = pd.read_csv(self._header_file).columns.values

        with context.log("load Data file: %s", self._data_file, logger=self._logger):
            df = pd.read_csv(self._data_file, names=headers, na_values=['\N', 'nan', 'None'], dtype=str, delimiter='$')

        # save the data for the reuested labels
        with context.log("loading labels data", logger=self._logger):
            self._labels_data = {
                label_name: df[label_name].fillna('0').astype(np.uint8).as_matrix().reshape((len(df), 1))
                for label_name in self._labels}

        with context.log("create manual feature combinations for location", logger=self._logger):
            df['country_state'] = pd.Series(
                np.where(df['exchange2'] == 'adx', df['loc_state'],
                         df['loc_country'].astype(str) + ':' + df['loc_state'].astype(str)))
            df['country_state_city'] = pd.Series(
                np.where(df['exchange2'] == 'adx', df['loc_city'],
                         df['country_state'].astype(str) + ':' + df['loc_city'].astype(str)))

        with context.log("create manual feature combination for hour of week", logger=self._logger):
            df['HOW'] = timeutils.hour_of_week_from_df(df.timestamp, df.timezone_offset.astype(str))

        with context.log("convert list features into one hot representation", logger=self._logger):
            list_features = transform.convert_list_features_to_one_hot(df,
                                                                       DATA_LIST_COLUMNS,
                                                                       self._value_rarity_threshold)
            list_features_matrix = list_features.matrix
            list_features_names = list_features.column_names

        with context.log("convert attribute features into one hot representation", logger=self._logger):
            attribute_features = transform.convert_attribute_features_to_one_hot(df,
                                                                                 DATA_ATTRIBUTE_COLUMNS,
                                                                                 self._value_rarity_threshold)
            attribute_features_matrix = attribute_features.matrix
            attribute_features_names = attribute_features.column_names

        # save time information, before deleting the full data frame
        self._day_ts = df['day_ts']
        self._timestamp = df['timestamp']

        # Combine the one-hot attribute and list features into a single structure
        with context.log('combine list and attribute features into a single final matrix', logger=self._logger):
            all_features_matrix = hstack([attribute_features_matrix, list_features_matrix],
                                         format='lil')  # concat the matrices horizontally
            all_features_names = attribute_features_names + list_features_names

        with context.log('patching location features', logger=self._logger):
            (self._features_data, self._feature_names) = \
                transform.patch_location_features(MatrixWithNames(all_features_matrix, all_features_names))

        self._logger.debug("Final one-hot features matrix shape: %d. %d", *self._features_data.shape)

        self._loaded = True

    def load(self, label):
        """
        A Builder for a specific data set (a single label)
        
        :param str label: the label to use in this data set
        :return: A data set reference context manager
        :rtype: DataSetReference
        """
        self._load()  # lazy load

        return DataSetReference(self._features_data, self._feature_names,
                                self._labels_data[label], label,
                                self._day_ts, self._timestamp, self._tmp_folder)
