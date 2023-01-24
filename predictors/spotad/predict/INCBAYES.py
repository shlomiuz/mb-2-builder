from zipfile import ZipFile
from collections import Counter
import uuid
import json
import numpy as np
# This function moved between modules from 0.18.1 to 0.19
try:
    from sklearn.utils.fixes import logsumexp
except:
    from scipy.special import logsumexp

RECENT_MODEL_WEIGHT = 0.5
HISTORICAL_MODEL_WEIGHT = 1.0 - RECENT_MODEL_WEIGHT
POSITIVE_THRESHOLD = 20


# ------------------- #
#  Utility functions  #
# ------------------- #

def isdir_in_zip(z, name):
    return any(x.startswith(name + "/") for x in z.namelist())


def argmax(l):
    return l.index(max(l))


# --------------- #
#  Model classes  #
# --------------- #

class BayesModel(object):
    def __init__(self, file_name, data_prefix):

        self._positive_counts = Counter()
        self._negative_counts = Counter()
        self._total_positive = None
        self._total_negative = None
        self._feature2index = dict()
        self._exists = True

        with ZipFile(file_name, 'r') as z:
            if not isdir_in_zip(z, data_prefix):
                self._exists = False
                return

            with z.open(data_prefix + "/positive_counts", 'r') as f:
                for line in map(str.strip, f):
                    if line == '': continue
                    cols = line.split('\t')
                    k = tuple(map(int, cols[0].split(',')))
                    c = int(cols[1])
                    self._positive_counts[k] = c
            with z.open(data_prefix + "/negative_counts", 'r') as f:
                for line in map(str.strip, f):
                    if line == '': continue
                    cols = line.split('\t')
                    k = tuple(map(int, cols[0].split(',')))
                    c = int(cols[1])
                    self._negative_counts[k] = c
            with z.open(data_prefix + "/label_counts", 'r') as f:
                self._total_positive = int(next(f).split('\t')[1])
                self._total_negative = int(next(f).split('\t')[1])
            with z.open("index", 'r') as f:
                for line in map(str.strip, f):
                    if line == '': continue
                    cols = line.split('\t')
                    self._feature2index[cols[0]] = int(cols[1])

        self._index2feature = {i: f for f, i in self._feature2index.iteritems()}
        self._all_index_combinations = list(set(self._positive_counts).union(set(self._negative_counts)))

        # Prepare likelihood cache (pre-compute the log likelihood sum as though all features are with value 0)
        # log(1 - P(x_1 | y)) + log(1 - P(x_2 | y)) + ...
        # P(x_i | y) is estimated with +1 smoothing as:  count(x_i, y) + 1 / count(y) + 2

        self._positive_log_likelihood_zero_total = np.sum((
            np.log(1.0 - ((self._positive_counts[fi] + 1.0) / (self._total_positive + 2.0)), dtype=np.float64)
            for fi in self._all_index_combinations
        ))

        self._negative_log_likelihood_zero_total = np.sum((
            np.log(1.0 - ((self._negative_counts[fi] + 1.0) / (self._total_negative + 2.0)), dtype=np.float64)
            for fi in self._all_index_combinations
        ))

        # Prepare the value that will need to be summed into the total to flip a sighting of 0 to a sighting of 1
        # for feature x_i this value represents the log(P(x_i |y) / (1 - P(x_i |y)))
        self._positive_log_likelihood_adjustment = {fi:
                                                        np.log((self._positive_counts[fi] + 1.0) / (
                                                            self._total_positive + 2.0), dtype=np.float64) -
                                                        np.log(1.0 - ((self._positive_counts[fi] + 1.0) / (
                                                            self._total_positive + 2.0)), dtype=np.float64)
                                                    for fi
                                                    in self._all_index_combinations}

        self._negative_log_likelihood_adjustment = {fi:
                                                        np.log((self._negative_counts[fi] + 1.0) / (
                                                            self._total_negative + 2.0), dtype=np.float64) -
                                                        np.log(1.0 - ((self._negative_counts[fi] + 1.0) / (
                                                            self._total_negative + 2.0)), dtype=np.float64)
                                                    for fi
                                                    in self._all_index_combinations}

        # log prior with +1 smoothing
        self._pos_log_prior = np.log(self._total_positive + 1.0, dtype=np.float64) - np.log(
            self._total_positive + self._total_negative + 2.0, dtype=np.float64)
        self._neg_log_prior = np.log(self._total_negative + 1.0, dtype=np.float64) - np.log(
            self._total_positive + self._total_negative + 2.0, dtype=np.float64)

    def exists(self):
        return self._exists

    @property
    def total_positive_counts(self):
        return self._total_positive

    def _indices_to_combination_indices(self, individual_indices):
        return [(x, y) for x in individual_indices for y in individual_indices if x <= y]

    def _features_to_combination_indices(self, feature_list):
        individual_indices = [self._feature2index[fn] for fn in feature_list if fn in self._feature2index]
        return self._indices_to_combination_indices(individual_indices)

    def _convert_to_combination_names(self, combination_indexes):
        return [self._index2feature[x] + '&' + self._index2feature[y]
                for (x, y) in combination_indexes
                if x in self._index2feature and y in self._index2feature]

    def predict_from_indices_combinations(self, combinations_list):
        pos_log_likelihood = np.sum((
            self._positive_log_likelihood_adjustment[fi]
            for fi in combinations_list
            if fi in self._positive_log_likelihood_adjustment
        )) + self._positive_log_likelihood_zero_total

        neg_log_likelihood = np.sum((
            self._negative_log_likelihood_adjustment[fi]
            for fi in combinations_list
            if fi in self._negative_log_likelihood_adjustment
        )) + self._negative_log_likelihood_zero_total

        # The denumenator is computed with sklearn utility function for numeric stability
        # ljls = logs of the joined likelihoods
        ljls = np.array([neg_log_likelihood + self._neg_log_prior, pos_log_likelihood + self._pos_log_prior],
                        dtype=np.float64)
        log_prob1 = ljls[1] - logsumexp(ljls)
        log_prob0 = ljls[0] - logsumexp(ljls)
        return np.exp(log_prob1), (1.0 if log_prob1 > log_prob0 else 0.0)

    def predict_from_indices(self, indices_list):
        combinations_list = self._indices_to_combination_indices(indices_list)
        return self.predict_from_indices_combinations(combinations_list)

    def predict_from_features(self, feature_list):
        combinations_list = self._features_to_combination_indices(feature_list)
        return self.predict_from_indices_combinations(combinations_list), self._convert_to_combination_names(
            combinations_list)

    def predict(self, feature_list):
        return self.predict_from_features(feature_list)[0]  # only return probability


class CombinedModel(object):
    def __init__(self, hist_model, recent_model):
        """
        
        :param BayesModel hist_model: 
        :param BayesModel recent_model: 
        """
        self._hist_model = hist_model
        self._recent_model = recent_model

    def predict(self, feature_list):
        hp, hf = self._hist_model.predict(feature_list)
        rp, rf = self._recent_model.predict(feature_list)

        return hp * HISTORICAL_MODEL_WEIGHT + rp * RECENT_MODEL_WEIGHT, list(set(hf + rf))


# -------------- #
#  RTB Interface #
# -------------- #

id2model = {}


def load(model_files):
    global pointers_count
    model_zip_file = model_files[0]

    historical_model = BayesModel(model_zip_file, 'historical')
    recent_model = BayesModel(model_zip_file, 'recent')

    if recent_model.exists() and recent_model.total_positive_counts >= POSITIVE_THRESHOLD and RECENT_MODEL_WEIGHT > 0.0:
        full_model = CombinedModel(historical_model, recent_model)
    else:
        full_model = historical_model

    model_id = str(uuid.uuid4())
    id2model[model_id] = full_model
    return model_id


def free(model_id):
    del id2model[model_id]
    return "free"


def predict(model_id, all_agents_props_list, bid_request_props, num_single_agent_props):
    model = id2model[model_id]
    num_agents = len(all_agents_props_list) / num_single_agent_props
    predictions = []
    attributes = []

    for agent_index in range(num_agents):
        agent_props = all_agents_props_list[agent_index: agent_index + num_single_agent_props]
        probability, features_used = model.predict(bid_request_props + agent_props)
        predictions.append(probability)
        attributes.append(features_used)

    highest_probability_index = argmax(predictions)

    return {
        "predictions": predictions,
        "attributes": attributes[highest_probability_index]
    }


def features():
    return json.dumps(
        {"request": [{"name": "data_center", "field": "dc_log"},
                     {"name": "impression_position_val", "field": "imp.0.banner.pos"},
                     {"name": "impression_viewability", "field": "unparseable.viewability"},
                     {"name": "ad_type", "field": "imp.0.imp_type", "type": 2,
                      "function": "var dynamicFeaturesFunc = function (name, value) { if (value == 'nativead') { return name + value; } else { return name + 'OTHER' } }"},
                     {"name": "formats", "field": "imp.0.banner", "type": 2,
                      "function": "var dynamicFeaturesFunc=function(keyWord,banner){var keys=[];if(banner.w.length>0&&banner.h.length>0){var h=banner.h;var w=banner.w;keyWord+='[';for(var i=0;i<h.length&&i<w.length;++i){keyWord+=+w[i]+'x'+h[i]+':';};keyWord[keyWord.length-1]=']';keys.push(keyWord);}else{keys.push(keyWord+banner.w+'x'+banner.h);};return keys;}"},
                     {"name": "app_site_ind", "field": "site", "const": "MWEB", "type": 3},
                     {"name": "publisher", "field": "site.publisher.id"},
                     {"name": "domain", "field": "site.page", "type": 2,
                      "function": "var dynamicFeaturesFunc=function(keyWord,page){return keyWord+funcs.extractDomain(page);}"},
                     {"name": "app_site_ind", "field": "app", "const": "INAPP", "type": 3},
                     {"name": "publisher", "field": "app.publisher.id"},
                     {"name": "domain", "field": "app.bundle", "type": 2,
                      "function": "var dynamicFeaturesFunc=function(keyWord,page){return keyWord+funcs.extractDomain(page);}"},
                     {"name": "location", "field": "device.geo.zip", "type": 2,
                      "function": "var dynamicFeaturesFunc = function (keyWord, zip) { return keyWord + zip.replace(/ /g, ''); }"},
                     {"name": "device_language", "field": "device.language"},
                     {"name": "browser", "field": "device.browser", "type": 3, "other": "Other"},
                     {"name": "device_make", "field": "device.make"},
                     {"name": "device_model", "field": "device.model"},
                     {"name": "device_os", "field": "device.os"},
                     {"name": "device_osv", "field": "device.osv"},
                     {"name": "connectiontype", "field": "device.connectiontype"},
                     {"name": "location", "field": "ext.locationSegments", "type": 1},
                     {"name": "category", "field": "site.cat", "type": 1},
                     {"name": "category", "field": "app.cat", "type": 1},
                     {"name": "HOW", "field": "device.ext.timezone_offset", "other_field": "timezone",
                      "type": 2,
                      "function": "var dynamicFeaturesFunc=function(keyWord,timezone_offset){var date=new Date();var hofw=date.getDay()*24+date.getHours();hofw=((hofw+168+timezone_offset)%168).toString();return keyWord+hofw;}"},
                     {"name": "TOD", "field": "device.ext.timezone_offset", "other_field": "timezone",
                      "type": 2,
                      "function": "var dynamicFeaturesFunc=function(keyWord,timezone_offset){var date=new Date();var dayTime=date.getHours();var tod=((dayTime + timezone_offset) % 24).toString();return keyWord+tod;}"}
                     ],
         "agent": [  # {"name": "banner", "field": "banner"}, - Banner ignored because video campaigns
             {"name": "sub_account", "field": "subAccount"},
             {"name": "banner_size", "field": "format"}
         ]
         }
    )


def validation():
    return json.dumps(
        {"agent_attributes": ["b", "g", "f"],  # assuming the counter for predict comes from the features definition
         "request_attributes": ["a"],
         "predict_result": [0.731951393853],
         "model": "000000000000"}
    )


def filestypes():
    return json.dumps(["model"])
