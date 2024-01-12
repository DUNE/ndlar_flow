################################################################################
##                                                                            ##
##    CONTAINS: Methods to parse files or objects and/or assist in new file/  ##
##              object creation e.g. hdf5 file parsing and json dictionary    ##
##              creation.                                                     ##
##                                                                            ##
################################################################################
import json 

####--------------------------- HDF5 FILE PARSING --------------------------####

def print_keys_attributes(data_h5):
    print('FILE KEYS:', list(data_h5.keys()),'\n')
    print('CHARGE KEYS:',list(data_h5['charge'].keys()),'\n')
    print('CHARGE EVENT DATA KEYS:',data_h5['charge']['events']['data'].dtype.names,'\n')
    print('CHARGE EVENT Q SIZE, DTYPE:',data_h5['charge']['events']['data']['q'].size,\
          ',',data_h5['charge']['events']['data']['q'].dtype,'\n')
    print('CHARGE EVENT NHIT SIZE, DTYPE:',data_h5['charge']['events']['data']['nhit'].size,\
          ',',data_h5['charge']['events']['data']['nhit'].dtype,'\n')
    print('CHARGE EVENT CHARGE REF KEYS:',list(data_h5['charge']['events']['ref']['charge'].keys()),'\n')
    print('CHARGE EVENT CHARGE REF HITS REF DTYPE:',\
          data_h5['charge']['events']['ref']['charge']['hits']['ref'].dtype,'\n')
    print('CHARGE HITS DATA KEYS:',data_h5['charge']['hits']['data'].dtype.names,'\n')
    print('COMBINED KEYS:',list(data_h5['combined'].keys()),'\n')
    print('GEOMETRY INFO KEYS:',list(data_h5['geometry_info'].keys()),'\n')
    print('LAR INFO KEYS:',list(data_h5['lar_info'].keys()),'\n')
    print('RUN INFO KEYS:',list(data_h5['run_info'].keys()),'\n')
    

def get_charge_datasets(data_h5):      

    events_data = data_h5['charge']['events']['data']
    hits_data = data_h5['charge']['hits']['data']

    return events_data, hits_data

####----------------------- OUTPUT DICTIONARY TO JSON ----------------------####

def tuple_key_to_string(d):
    out={}
    for key in d.keys():
        string_key=""
        max_length=len(key)
        for i in range(max_length):
            if i<len(key)-1: string_key+=str(key[i])+"-"
            else: string_key+=str(key[i])
        out[string_key]=d[key]
    return out                    
            

def save_dict_to_json(d, name, if_tuple):
    with open(name+".json", "w") as outfile:
        if if_tuple==True:
            updated_d = tuple_key_to_string(d)
            json.dump(updated_d, outfile, indent=4)
        else:
            json.dump(d, outfile, indent=4)    