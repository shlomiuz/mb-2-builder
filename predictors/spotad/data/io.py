from pathlib2 import Path
import gzip
from operator import methodcaller
from scipy.sparse import coo_matrix
import numpy as np
from spotad.data.common import MultiLabelDataSet
from functools import reduce
import logging

PARTS_FILE_PATTERN = "part-*"

logger = logging.getLogger('sparse.load_data_set')


def load_header(header_path,
                delimiter='\t'):
    """
    Load only the header information
    
    :param str|pathlib2.Path header_path: the hadoop folder where the header resides
    :param str delimiter: the delimiter used in the input sparse files
    :return: the list of columns in the header
    :rtype: list[str]
    """
    if isinstance(header_path, str):
        header_path = Path(header_path)

    # check that both exist and are folders, and contain valid part files
    _validate_path(header_path, expected_parts=1)

    # read in the header
    header_file = next(header_path.glob(PARTS_FILE_PATTERN))

    # determine compression based on the extension

    with _open_read_smart(header_file) as f:
        index2column = next(f).strip().split(delimiter)

    return index2column


def load_sparse_data_set(header_path,
                         data_path,
                         delimiter='\t',
                         sparse_output_format='csr',
                         labels={'click', 'download', 'buy', 'register', 'video_ad_start'}):
    """
    Load data set produced by ds-transformer sparse format, can automatically handle gzipped files based
      on the extension.
    
    :param str|pathlib2.Path header_path: the hadoop folder where the header resides
    :param str|pathlib2.Path data_path: the hadoop folder where partitions of the data reside
    :param str delimiter: the delimiter used in the input sparse files
    :param str sparse_output_format: the scipy format of the data sparse matrix
    :param set labels: what labels to consider, all others will be assumed as data (features) 
    :return: all the data set information, including header, sparse data matrix, and all label vectors
    :rtype: spotad.io.MultiLabelDataSet
    """
    if isinstance(header_path, str):
        header_path = Path(header_path)

    if isinstance(data_path, str):
        data_path = Path(data_path)

    # check that both exist and are folders, and contain valid part files
    _validate_path(header_path, expected_parts=1)
    _validate_path(data_path)

    # read in the header
    header_file = next(header_path.glob(PARTS_FILE_PATTERN))

    # determine compression based on the extension

    with _open_read_smart(header_file) as f:
        index2column = next(f).strip().split(delimiter)

    data2index = {c: i for i, c in enumerate(index2column) if c not in labels}
    label2index = {c: i for i, c in enumerate(index2column) if c in labels}

    # ASSUMPTION: label indexes come after the data indexes
    last_data_index = reduce(lambda mx, v: max(mx, v), data2index.values())

    # read the files that match the pattern: part-r-(00000) - 5 digits and sort by part
    data_files = sorted(list(data_path.glob(PARTS_FILE_PATTERN)))

    # read data as coordinates ready for the coo format
    row_index = 0
    data_coordinates = list()
    ys = dict(map(lambda label: (label, []), labels))
    for data_path in data_files:

        logger.debug('Loading: %s', data_path)

        with _open_read_smart(data_path) as f:
            for line in map(methodcaller('strip'), f):
                try:
                    col_indexes = map(int, line.split(delimiter))
                    data_indexes, label_indexes = _divide_list_values(lambda x: x <= last_data_index, col_indexes)

                    # get the data information
                    data_row_coordinates = [(row_index, c) for c in data_indexes]
                    data_coordinates.extend(data_row_coordinates)

                    # get the labels information
                    for label, index in label2index.iteritems():
                        ys[label].append(1 if index in label_indexes else 0)

                    row_index += 1
                except:
                    print("row: {} is broken".format(row_index))
                    continue

    num_rows = row_index

    # create scipy sparse data frame from the coordinates
    row, col = map(lambda m: np.array(m, dtype=np.uint32), zip(*data_coordinates))  # unzip
    data = np.ones(row.shape)

    data_shape = (num_rows, last_data_index + 1)
    sparse_data_matrix = coo_matrix((data, (row, col)), data_shape)

    # convert to column sparse format
    final_data_matrix = sparse_data_matrix.asformat(sparse_output_format)
    # convert labels into column vectors
    # ys = {k: np.array(v, dtype=np.uint8).reshape((num_rows, 1)) for k, v in ys.iteritems()}
    ys = {k: np.array(v, dtype=np.uint8) for k, v in ys.iteritems()}

    return MultiLabelDataSet(index2column[:last_data_index + 1], final_data_matrix, ys)


def _divide_list_values(f, lst):
    """
    Divide the values in a lst into two lists, based on the provided predicate
    
    :param callable f: the predicate, should return 'true' to place value in the first list, 'false' for second list
    :param list lst: the list to divide
    :return: two disjoint lists, each containing different values from the original list
    :rtype: (list,list)
    """
    return reduce(lambda acc, val: (acc[0] + [val], acc[1]) if f(val) else (acc[0], acc[1] + [val]), lst, ([], []))


def _validate_path(path, expected_parts=None):
    """
    Validate that the path is a valid directory containing hadoop type file partitions
    It wil raise IO Error in case some problem is detected
    
    :param Path path: the path to validate
    """
    if not path.exists(): raise IOError("Path does not exist: {}".format(path.as_posix()))
    if not path.is_dir(): raise IOError("Path is not a directory: {}".format(path.as_posix()))
    part_files = list(path.glob(PARTS_FILE_PATTERN))
    if len(part_files) == 0:
        raise IOError("No files of form {0} found inside: {1}".format(PARTS_FILE_PATTERN, path.as_posix()))
    if expected_parts is not None:
        if len(part_files) != expected_parts:
            raise IOError("Amount of parts doesn't match what was expected. actual: {0}, expected:{1}, {2}"
                          .format(len(part_files), expected_parts, path.as_posix()))


def _open_read_smart(path):
    """
    Acts as regular open with read option that can 
    automatically distinguish between regular and gzipped files based on file extension
    
    :param path: the path to open
    :return: the file handler
    """

    is_compressed = path.as_posix().endswith(".gz")
    if not is_compressed:
        return open(path.as_posix(), 'r')
    else:
        return gzip.open(path.as_posix(), 'rb')
