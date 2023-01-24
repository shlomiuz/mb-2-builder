import logging
from scipy import sparse
import numpy as np
from spotad.data.common import AbstractMultiLabelDataGroup
from spotad.transform import MatrixWithNames
import os
import uuid
from spotad import context
import subprocess
import gzip

_COMBINE_DATA_SETS_COMMAND_TEMPLATE = "zcat {test_file} | tail -n +2 | gzip >> {train_file}"
_COMBINE_DATA_CASE_WEIGHTS_COMMAND_TEMPLATE = "cat {test_file} >> {train_file}"
_CASE_WEIGHTS_FILE_POSTFIX = '.case_weights'

mlds_logger = logging.getLogger("RangerMultiLabelDataSet")


class RangerMultiLabelDataGroup(AbstractMultiLabelDataGroup):
    def __init__(self, named_multi_sets, tmp_folder):
        super(RangerMultiLabelDataGroup, self).__init__(named_multi_sets)
        self._tmp_folder = tmp_folder

    def focus_on_set(self, label):
        return RangerDataGroup({name: mlds.focus_on_set(label) for name, mlds in self._named_multi_sets.iteritems()},
                               self._tmp_folder)


def _write_named_matrix(data, file):
    """
    Write a ranger compatible CSV file from a matrix and a list of column names
    
    :param MatrixWithNames data: values matrix and the names of the columns
    :param str file_name: path of the CSV file to write
    """

    with gzip.open(file, 'w') as out:
        # write header manually, numpy header writes header as a comment
        out.write('\t'.join(data.column_names))
        out.write('\n')

        # write the data
        for i in range(data.matrix.shape[0]):
            out.write('\t'.join(map(str, data.matrix.getrowview(i).todense().astype(np.uint8).tolist()[0])))
            out.write('\n')

def _write_case_weights(data, file):
    with open(file, 'w') as out:
        np.savetxt(out, data.reshape((1, data.shape[0])), fmt='%1.5f', delimiter=' ', newline=' ')


def _compute_case_weights(y):
    num_samples = y.shape[0]
    return tuple(float(num_samples) / (2 * np.bincount(y.reshape(num_samples))))


rdg_logger = logging.getLogger("RangerDataGroup")


class RangerDataGroup(object):
    def __init__(self, named_sets, tmp_folder):
        self._named_sets = named_sets
        self._realized_named_sets = None
        self._in_context = False
        self._tmp_folder = tmp_folder

    def __getitem__(self, name):
        if self._realized_named_sets is None:
            raise AssertionError("data set accessed outside a with block")
        else:
            return self._realized_named_sets[name]

    @property
    def names(self):
        return self._named_sets.keys()

    @property
    def header(self):
        return self._named_sets.values()[0].header

    def __enter__(self):

        # Generate files from the data, in a form that ranger is able to read from
        self._realized_named_sets = dict()


        # TODO - can we write multiple sets in parallel and get any speed benefit?
        for name in self.names:

            X = self._named_sets[name].X
            y = self._named_sets[name].y
            header = self._named_sets[name].header

            negative_weight, positive_weight = _compute_case_weights(y)
            case_weights = np.ones(y.shape[0]) * negative_weight
            case_weights[y.reshape(y.shape[0]) == 1] = positive_weight

            matrix = MatrixWithNames(
                sparse.hstack([X, y.reshape(len(y), 1)], format='lil'),  # convert y to column vector before hstack
                header + ['__label__']
            )

            # Prepare a tmp folder if it doesnt exists
            if not self._tmp_folder.exists():
                os.makedirs(self._tmp_folder.as_posix())

            # generate unique file names
            unique_id = uuid.uuid4()
            data_file = (self._tmp_folder / '{0}-{1}.csv.gz'.format(name, unique_id)).as_posix()
            with context.log("Writing %s set temporary file: %s", name, data_file,
                             logger=rdg_logger):
                _write_named_matrix(matrix, data_file)

            case_weights_file = data_file + _CASE_WEIGHTS_FILE_POSTFIX
            with context.log("Writing %s set case weights temporary file: %s", name, case_weights_file,
                             logger=rdg_logger):
                _write_case_weights(case_weights, case_weights_file)

            self._realized_named_sets[name] = RangerDataSet(header, y, data_file, case_weights_file)

        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_context = False

        for name, ds in self._realized_named_sets.iteritems():
            rdg_logger.info("Deleting temporary files of set %s", name)
            del ds

        self._realized_named_sets = None
        return True # Don't swallow any exceptions

    def unified_set(self):
        return RangerCombinedDataSet(self)

rds_logger = logging.getLogger("RangerDataSet")
class RangerDataSet(object):
    def __init__(self, header, gold_labels, data_file, case_weights_file):
        self._header = header
        self._gold_labels = gold_labels
        self._data_file = data_file
        self._case_weights_file = case_weights_file
        self._valid = True

    def _validate(self, prop_name):
        if not self._valid:
            raise AssertionError("property '{}' accessed outside a with block".format(prop_name))

    @property
    def header(self):
        return self._header

    @property
    def data_file(self):
        self._validate('data_file')
        return self._data_file

    @property
    def case_weights_file(self):
        self._validate('case_weights_file')
        return self._case_weights_file

    @property
    def label_name(self):
        return '__label__'

    @property
    def gold_labels(self):
        self._validate('true_labels')
        return self._gold_labels

    @property
    def y(self):
        self._validate('true_labels')
        return self._gold_labels

    def invalidate(self):
        self._valid = False

    def __del__(self):
        rds_logger.info("Deleting files: %s, %s",
                        self.data_file,
                        self._case_weights_file)
        os.remove(self._data_file)
        os.remove(self._case_weights_file)
        self._gold_labels = None


rcds_logger = logging.getLogger('RangerCombinedDataSet')

class RangerCombinedDataSet(object):
    def __init__(self, data_group):
        self._in_context = False
        self._data_group = data_group

    def _validate_in_context(self, prop_name):
        if not self._in_context:
            raise AssertionError("property '{}' accessed outside a with block".format(prop_name))

    @property
    def data_file(self):
        self._validate_in_context('data_file')
        # For speed, the combination appends the test into the exists train set
        return self._data_group['train'].data_file

    @property
    def case_weights_file(self):
        self._validate_in_context('case_weights_file')
        # For speed, the combination appends the test into the exists train set
        return self._data_group['train'].case_weights_file


    @property
    def label_name(self):
        return '__label__'

    @property
    def header(self):
        self._validate_in_context('header')
        return self._data_group.header

    def __enter__(self):
        # append the test set (without the header to the existing train set)
        combine_set_command = _COMBINE_DATA_SETS_COMMAND_TEMPLATE.format(test_file=self._data_group['test'].data_file,
                                                                         train_file=self._data_group['train'].data_file)

        combine_case_weights_command = _COMBINE_DATA_CASE_WEIGHTS_COMMAND_TEMPLATE.format(
            test_file=self._data_group['test'].case_weights_file,
            train_file=self._data_group['train'].case_weights_file)

        with context.log("Appending test set to train, to produce a full data set", logger=rcds_logger):
            rcds_logger.debug('executing: %s', combine_set_command)
            status = subprocess.call(combine_set_command, shell=True)  # working directory
            if status != 0:
                raise Exception("Failed creating a full data set")

            rcds_logger.debug('executing: %s', combine_case_weights_command)
            status = subprocess.call(combine_case_weights_command, shell=True)  # working directory
            if status != 0:
                raise Exception("Failed creating a full data set case weights")

        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # There is nothing to actually delete, we did not create any new files so the group will be in charge
        # of deleting the temporary files
        self._in_context = False
        return True # Don't swallow any exceptions
