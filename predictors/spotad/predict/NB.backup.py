from scipy.sparse import *
import numpy as np

pointers = {}
pointers_count = 0


def validation():
    return "{\"agent_attributes\":[\"banner26613a2518b418ede5ee3e942b30bfa3c3da257344e2bee9876c11e9b8c99d07_7333\",\"sub_account275\",\"banner_size0x0\"],\"request_attributes\":[\"data_centerEUROPE\",\"impression_position_val1\",\"ad_typeOTHER\",\"formats300x250\",\"app_site_indINAPP\",\"publisher11726\",\"domain372648912.com\",\"location32401\",\"device_languageen\",\"browserOther\",\"device_makeApple\",\"device_modeliPhone\",\"device_osiOS\",\"device_osv10.0\",\"connectiontype2\",\"locationUSA\",\"locationUSA:FL\",\"locationUSA:FL:Navarre\",\"categoryIAB1\",\"categoryIAB24\",\"HOW40\"],\"predict_result\":[6.041387897249942e-05],\"model\":\"149916137184\"}"


def load(model, keymap):
    global pointers_count
    pointers_count += 1
    pointers[str(pointers_count)] = np.load(model)[()]
    return str(pointers_count)


def free(pointer):
    del pointers[pointer]
    return "free"


def predict(pointer, ag, br, counter):
    keymap = pointers[pointer]['keymap']
    model = pointers[pointer]['model']
    resp = {}
    agents = len(ag) / counter
    attr = []
    x = lil_matrix((agents, len(keymap)))
    for row in np.arange(agents):
        data = np.concatenate((ag[(row * counter):(row * counter + counter)], br), axis=0)
        indices, attr0 = zip(*[(keymap[data_item], data_item) for data_item in data if data_item in keymap])
        attr.append(attr0)
        x[row, indices] = 1
    resp['predictions'] = model.predict_proba(x)[:, 1].tolist()
    attr = np.array(attr)
    resp['attributes'] = attr[resp['predictions'] == np.max(resp['predictions'])][0].tolist()
    return resp


def features():
    return "{\"request\":[{\"name\":\"data_center\",\"field\":\"dc_log\"},{\"name\":\"impression_position_val\",\"field\":\"imp.0.banner.pos\"},{\"name\":\"impression_viewability\",\"field\":\"unparseable.viewability\"},{\"name\":\"ad_type\",\"field\":\"imp.0.imp_type\",\"type\":2,\"function\":\"var dynamicFeaturesFunc = function (name, value) { if (value == 'nativead') { return name + value; } else { return name + 'OTHER' } }\"},{\"name\":\"formats\",\"field\":\"imp.0.banner\",\"type\":2,\"function\":\"var dynamicFeaturesFunc=function(keyWord,banner){var keys=[];if(banner.w.length>0&&banner.h.length>0){var h=banner.h;var w=banner.w;keyWord+='[';for(var i=0;i<h.length&&i<w.length;++i){keyWord+=+w[i]+'x'+h[i]+':';};keyWord[keyWord.length-1]=']';keys.push(keyWord);}else{keys.push(keyWord+banner.w+'x'+banner.h);};return keys;}\"},{\"name\":\"app_site_ind\",\"field\":\"site\",\"const\":\"MWEB\",\"type\":3},{\"name\":\"publisher\",\"field\":\"site.publisher.id\"},{\"name\":\"domain\",\"field\":\"site.page\",\"type\":2,\"function\":\"var dynamicFeaturesFunc=function(keyWord,page){return keyWord+funcs.extractDomain(page);}\"},{\"name\":\"app_site_ind\",\"field\":\"app\",\"const\":\"INAPP\",\"type\":3},{\"name\":\"publisher\",\"field\":\"app.publisher.id\"},{\"name\":\"domain\",\"field\":\"app.bundle\",\"type\":2,\"function\":\"var dynamicFeaturesFunc=function(keyWord,page){return keyWord+funcs.extractDomain(page);}\"},{\"name\":\"location\",\"field\":\"device.geo.zip\",\"type\":2,\"function\":\"var dynamicFeaturesFunc = function (keyWord, zip) { return keyWord + zip.replace(/ /g, ''); }\"},{\"name\":\"device_language\",\"field\":\"device.language\"},{\"name\":\"browser\",\"field\":\"device.browser\",\"type\":3,\"other\":\"Other\"},{\"name\":\"device_make\",\"field\":\"device.make\"},{\"name\":\"device_model\",\"field\":\"device.model\"},{\"name\":\"device_os\",\"field\":\"device.os\"},{\"name\":\"device_osv\",\"field\":\"device.osv\"},{\"name\":\"connectiontype\",\"field\":\"device.connectiontype\"},{\"name\":\"location\",\"field\":\"ext.locationSegments\",\"type\":1},{\"name\":\"category\",\"field\":\"site.cat\",\"type\":1},{\"name\":\"category\",\"field\":\"app.cat\",\"type\":1},{\"name\":\"HOW\",\"field\":\"device.ext.timezone_offset\",\"other_field\":\"timezone\",\"type\":2,\"function\":\"var dynamicFeaturesFunc=function(keyWord,timezone_offset){var date=new Date();var hofw=date.getDay()*24+date.getHours();hofw=((hofw+168+timezone_offset)%168).toString();return keyWord+hofw;}\"}],\"agent\":[{\"name\":\"banner\",\"field\":\"banner\"},{\"name\":\"sub_account\",\"field\":\"subAccount\"},{\"name\":\"banner_size\",\"field\":\"format\"}]}"
