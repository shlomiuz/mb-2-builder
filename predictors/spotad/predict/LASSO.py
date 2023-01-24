from scipy.sparse import *
import numpy as np

pointers = {}
pointers_count = 0


def validation():
    return "{\"agent_attributes\":[\"bannerc6c12bc3cf7b9239ee3730314d354238352e43e5f80149af24d45f875b4feed5_13456\"],\"request_attributes\":[\"datacenterUS_WEST\",\"impression_position_val0\",\"app_site_indMWEB\",\"publisher9933\",\"domainwww.usatoday.com\",\"locationUSA:CA:Ventura\",\"browserMobile Safari\",\"device_modelSwift 2 Plus\",\"categoryutilities\",\"HOW108\"],\"predict_result\":[0],\"model\":\"15003657773\"}"


def load(model, keymap):
    global pointers_count
    pointers_count += 1
    pointers[str(pointers_count)] = np.load(model)[()]
    return str(pointers_count)


def free(pointer):
    del pointers[pointer]
    return "free"


def predict(pointer, ag, br, counter):
    resp = {}
    keymap = pointers[pointer]['keymap']
    model = pointers[pointer]['model']
    confusion = pointers[pointer]['confusion']
    agents = len(ag) / counter
    attr = []
    x = lil_matrix((agents, len(keymap)))
    for row in np.arange(agents):
        data = np.concatenate((ag[(row * counter):(row * counter + counter)], br), axis=0)
        indices, attr0 = zip(*[(keymap[data_item], data_item) for data_item in data if data_item in keymap])
        attr.append(list(attr0))
        x[row, indices] = 1
    pred = model.predict(x)
    predictions = np.ones(pred.shape) * confusion['false_negative']
    predictions[pred == 1] = confusion['true_positive']
    resp['attributes'] = attr[np.argmax(predictions)]
    resp['predictions'] = predictions.tolist()
    attr = np.array(attr)
    resp['attributes'] = attr[resp['predictions'] == np.max(resp['predictions'])][0].tolist()
    return resp


def features():
    return "{\"request\":[{\"name\":\"data_center\",\"field\":\"dc_log\"},{\"name\":\"impression_position_val\",\"field\":\"imp.0.banner.pos\"},{\"name\":\"impression_viewability\",\"field\":\"unparseable.viewability\"},{\"name\":\"ad_type\",\"field\":\"imp.0.imp_type\",\"type\":2,\"function\":\"var dynamicFeaturesFunc = function (name, value) { if (value == 'nativead') { return name + value; } else { return name + 'OTHER' } }\"},{\"name\":\"formats\",\"field\":\"imp.0.banner\",\"type\":2,\"function\":\"var dynamicFeaturesFunc=function(keyWord,banner){var keys=[];if(banner.w.length>0&&banner.h.length>0){var h=banner.h;var w=banner.w;keyWord+='[';for(var i=0;i<h.length&&i<w.length;++i){keyWord+=+w[i]+'x'+h[i]+':';};keyWord[keyWord.length-1]=']';keys.push(keyWord);}else{keys.push(keyWord+banner.w+'x'+banner.h);};return keys;}\"},{\"name\":\"app_site_ind\",\"field\":\"site\",\"const\":\"MWEB\",\"type\":3},{\"name\":\"publisher\",\"field\":\"site.publisher.id\"},{\"name\":\"domain\",\"field\":\"site.page\",\"type\":2,\"function\":\"var dynamicFeaturesFunc=function(keyWord,page){return keyWord+funcs.extractDomain(page);}\"},{\"name\":\"app_site_ind\",\"field\":\"app\",\"const\":\"INAPP\",\"type\":3},{\"name\":\"publisher\",\"field\":\"app.publisher.id\"},{\"name\":\"domain\",\"field\":\"app.bundle\",\"type\":2,\"function\":\"var dynamicFeaturesFunc=function(keyWord,page){return keyWord+funcs.extractDomain(page);}\"},{\"name\":\"location\",\"field\":\"device.geo.zip\",\"type\":2,\"function\":\"var dynamicFeaturesFunc = function (keyWord, zip) { return keyWord + zip.replace(/ /g, ''); }\"},{\"name\":\"device_language\",\"field\":\"device.language\"},{\"name\":\"browser\",\"field\":\"device.browser\",\"type\":3,\"other\":\"Other\"},{\"name\":\"device_make\",\"field\":\"device.make\"},{\"name\":\"device_model\",\"field\":\"device.model\"},{\"name\":\"device_os\",\"field\":\"device.os\"},{\"name\":\"device_osv\",\"field\":\"device.osv\"},{\"name\":\"connectiontype\",\"field\":\"device.connectiontype\"},{\"name\":\"location\",\"field\":\"ext.locationSegments\",\"type\":1},{\"name\":\"category\",\"field\":\"site.cat\",\"type\":1},{\"name\":\"category\",\"field\":\"app.cat\",\"type\":1},{\"name\":\"HOW\",\"field\":\"device.ext.timezone_offset\",\"other_field\":\"timezone\",\"type\":2,\"function\":\"var dynamicFeaturesFunc=function(keyWord,timezone_offset){var date=new Date();var hofw=date.getDay()*24+date.getHours();hofw=((hofw+168+timezone_offset)%168).toString();return keyWord+hofw;}\"}],\"agent\":[{\"name\":\"banner\",\"field\":\"banner\",\"type\":3,\"other\":\"Other\"},{\"name\":\"sub_account\",\"field\":\"subAccount\"},{\"name\":\"banner_size\",\"field\":\"format\"}]}"
