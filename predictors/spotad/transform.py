from collections import namedtuple
import logging
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import psutil
from collections import Counter
from scipy.sparse import hstack

MatrixWithNames = namedtuple("MatrixWithNames", ['matrix','column_names'])


# combine values that appear less than some rarity threshold
def _combine_rare_values(df, column_names, rarity_threshold, convert_nan_to_other=True):
    """
    Combine all the values that appear less than the provided rarity threshold under a single
    value called OTHER. In addition remove columns that don't hold enough unique values.
    
    :param pandas.DataFrame df: the data frame to transform 
    :param int rarity_threshold: the rarity threshold
    :param boolean convert_nan_to_other:  should null values also be converted into the OTHER value
    """

    logger = logging.getLogger('transform._combine_rare_values')

    total_unique_values_before = 0
    total_unique_values_after = 0

    for col in column_names:
        levels_before = len(df[col].unique())
        total_unique_values_before += levels_before

        logger.debug('%s: Number of unique values BEFORE combinations=%d',
                     col, levels_before)

        counts = df[col].value_counts()

        indices_of_rare_values = df[col].isin(counts[counts < rarity_threshold].index)

        df.loc[indices_of_rare_values,col] = 'OTHER'

        if convert_nan_to_other:
            logger.debug('%s: Number of na\'s=%d',
                     col, df[col].isnull().sum())
            df.loc[:, col] = df[col].fillna('OTHER')

        levels_after = len(df[col].unique())
        total_unique_values_after += levels_after

        logger.debug('%s: Number of unique values AFTER combination: %d ', col, levels_after)
        if levels_after <= 1:
            del df[col]
            logger.debug('Column was removed from lack of values')

    logger.debug('Total number of unique values before combinations=%d, ' +
                 'Total number of unique values after combinations=%d',
                 total_unique_values_before, total_unique_values_after)

def convert_attribute_features_to_one_hot(df, column_names, rare_values_threshold=None):
    """
    Convert a data frame columns into a one hot representation matrix.
    
    :param pandas.DataFrame df: the data frame to transform
    :param list column_names: the name of the columns to attempt and change into one-hot representation
    :param int rare_values_threshold: the threshold for which to filter out rare values
    :return: a transformed matrix with the new names of the columns
    :rtype: MatrixWithName
    """
    logger = logging.getLogger('convert_attribute_features_to_one_hot')

    _combine_rare_values(df, column_names, rarity_threshold=rare_values_threshold)

    # change the data frame index to only contain the columns we are interested in, before pivoting
    remaining_columns = list(set(column_names).intersection(set(df.columns.values)))
    df = df[remaining_columns]

    ### transform attribute features to one hot vectors
    dv = DictVectorizer(sparse=True, separator="",dtype=np.uint8)
    x_columns = dv.fit_transform(df.to_dict(orient='records'))

    logger.debug("Attribute features matrix shape %d,%d",*x_columns.shape)

    logger.debug('Deleting attribute data frame')
    logger.debug("Virtual memory BEFORE deleting attribute features data frame: %s", psutil.virtual_memory())
    del df
    logger.debug("Virtual memory AFTER deleting attribute features data frame: %s", psutil.virtual_memory())

    ###feature names
    # remove 'OTHER' in columns
    indices_with_value_other = [i for i, s in enumerate(dv.feature_names_) if s.endswith('OTHER')]
    indices_without_value_other = list(set(np.arange(len(dv.feature_names_))).difference(set(indices_with_value_other)))
    x_columns = x_columns[:, indices_without_value_other]
    logger.debug("Attribute one-hot features matrix after removing the OTHER values shape: %d, %d",*x_columns.shape)
    feature_names_columns = list(np.asarray(dv.feature_names_)[indices_without_value_other])

    return MatrixWithNames(
        x_columns,
        feature_names_columns
    )

def convert_list_features_to_one_hot(df, column_names, rare_values_threshold=None):
    """
    Convert a data frame columns where each value represents a list of values separated by a dot(.)
     into a one hot representation matrix.

    :param pandas.DataFrame df: The data frame to transform
    :param list column_names: The name of the columns (assumed . separator in the column values)
    :param int rare_values_threshold: the threshold for which to filter out rare values
    :return: A Matrix with column names
    :rtype: MatrixWithName
    """
    logger = logging.getLogger('transform.convert_list_features_to_one_hot')
    matrices = []
    names = []

    for col in column_names:

        df[col] = df[col].fillna('OTHER')

        values_map = list(map(lambda x: {k: 1 for k in x.split(".")}, df[col]))
        vectorizer = DictVectorizer(sparse=True, separator="", dtype=np.uint8)
        one_hot_matrix = vectorizer.fit_transform(values_map)
        logger.debug("feature %s matrix shape: %d, %d", col, *one_hot_matrix.shape)

        # discard categories that don't appear more than the rarity threshold
        col_sums = np.asarray(one_hot_matrix.sum(axis=0))[0]
        one_hot_matrix = one_hot_matrix[:, col_sums >= rare_values_threshold]
        logger.debug("feature %s matrix shape: %d, %d, after discarding rare values", col, *one_hot_matrix.shape)

        # create unique names from the category one-hot features
        column_names = np.asarray(vectorizer.feature_names_)[col_sums >= rare_values_threshold]
        column_names = ["{0}{1}".format(col, name) for name in column_names]

        # remove 'OTHER' in category
        indices_with_value_other = [i for i, s in enumerate(column_names) if s.endswith('OTHER')]
        indices_without_value_other = list(set(np.arange(len(column_names))).difference(set(indices_with_value_other)))
        one_hot_matrix = one_hot_matrix[:, indices_without_value_other]
        logger.debug("feature %s one-hot matrix after removing the OTHER values shape: %d, %d", col, *one_hot_matrix.shape)
        one_hot_feature_names = list(np.asarray(column_names)[indices_without_value_other])

        matrices.append(one_hot_matrix.tolil())#tocsr())
        names.append(one_hot_feature_names)

    if len(matrices) > 1:
        return MatrixWithNames(
            hstack(matrices, format='lil'), # sparse hstack
            reduce(lambda acc,val: acc+val, names, [])
        )
    else:
        return MatrixWithNames(matrices[0], names[0])

def patch_location_features(matrix_with_names):
    """
    Patches location features
    
    :param MatrixWithNames matrix_and_names: a matrix with names for the columns
    :return: patched location features 
    :rtype: MatrixWithNames
    """
    logger = logging.getLogger('transform.patch_location_features')

    logger.debug("Number of features before patching %d", len(matrix_with_names.column_names))

    # replace geo feature names with 'location'
    adjusted_names = matrix_with_names.column_names
    for option in ['country_state_city', 'country_state', 'loc_country', 'zip']:
        adjusted_names = list(map(lambda n: n.replace(option,'location'), matrix_with_names.column_names))

    # remove duplicated keys
    duplicated_feature_names = [k for k, count in Counter(adjusted_names).iteritems() if count > 1]
    all_feature_names = np.asarray(adjusted_names)
    not_duplicated_indexes = ~np.in1d(all_feature_names, duplicated_feature_names)
    adjusted_names = list(all_feature_names[not_duplicated_indexes])
    logger.debug("Number of features after patching %d", len(adjusted_names))

    # remove from tha matrix
    return MatrixWithNames(
        matrix_with_names.matrix.tolil()[:, not_duplicated_indexes],
        adjusted_names)
