from spotad.data.io import load_header
from spotad.data.common import AbstractMultiLabelDataGroup


class Poly2BayesData(AbstractMultiLabelDataGroup):
    @staticmethod
    def load(header_path, train_path, train_data, test_path, test_data, tmp_folder):
        """
        
        :param pathlib2.Path header_path:
        :param pathlib2.Path train_path: 
        :param spotad.io.MultiLabelDataSet train_data:
        :param pathlib2.Path test_path: 
        :param spotad.io.MultiLabelDataSet test_data:
        :return: 
        :rtype: Poly2BayesData
        """
        columns = load_header(header_path)

        ALL_POSSIBLE_LABELS = {"click", "download", "buy", "register", "video_ad_start", "video_start"}
        available_labels = ALL_POSSIBLE_LABELS.intersection(columns)
        data_paths = {"train": train_path, "test": test_path}
        data = {"train": train_data, "test": test_data}

        return Poly2BayesData(header_path, list(available_labels), data_paths, data, tmp_folder)

    def __init__(self, header_path, labels, data_paths, data, tmp_folder, focused_set=None):

        super(Poly2BayesData, self).__init__(data)

        # gather available labels from the header
        self._header_path = header_path
        self._labels = labels
        self._data_paths = data_paths
        self._data = data
        self._tmp_folder = tmp_folder
        self._focused_set = focused_set

        # not actually used for anything in this implementation
        self._in_context = False

    def focus_on_set(self, label):
        if label not in self._labels:
            raise AssertionError("the label '{}' is not available in the data".format(label))
        else:
            return Poly2BayesData(self._header_path, [label], self._data_paths, self._data, self._tmp_folder)

    # override the AbstractMultiLabelDataGroup implementation
    # we avoid the regular splitting of data to keep all the information available,
    # simplifying the re-unification of test and train later on, the logical slicing is done by modifying the
    # 'focused_set' field
    def __getitem__(self, name):
        if name not in self._data_paths:
            raise AssertionError("the data set '{}' does not exists".format(name))
        else:
            return Poly2BayesData(self._header_path, self._labels, self._data_paths, self._data, self._tmp_folder,
                                  focused_set=name)

    @property
    def header_path(self):
        """
        
        :return: 
        :rtype: str
        """
        return self._header_path.as_posix()

    @property
    def data_path(self):
        """
        
        :return: 
        :rtype: str
        """
        if self._focused_set is not None:
            return self._data_paths[self._focused_set]
        else:
            return ":".join(map(lambda p: p.as_posix(), self._data_paths.values()))

    @property
    def goal(self):
        """
        
        :return: 
        :rtype: str
        """
        if len(self._labels) != 1:
            raise AssertionError(
                "The data is not focused on a single label, to be considered a goal: {}".format(self._labels))
        else:
            return self._labels[0]

    @property
    def y(self):
       return self._data[self._focused_set].ys[self.goal]

    def get_focused_data(self):
        """
        
        :return: 
        :rtype: MultiLabelDataSet
        """

        if (self._focused_set is not None):
            return self._data[self._focused_set]
        else:
            raise AssertionError("Tried accessing data of muti-set")

    def __enter__(self):
        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_context = False

    # unification is achieved by simply removing the focus, which means both train and test will be returned from
    # a 'data_path' call
    def unified_set(self):
        return Poly2BayesData(self._header_path, self._labels, self._data_paths, self._data, self._tmp_folder,
                              focused_set=None)
