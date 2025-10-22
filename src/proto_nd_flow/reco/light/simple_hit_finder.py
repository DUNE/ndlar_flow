import numpy as np
import numpy.ma as ma

from h5flow.core import H5FlowStage, resources

import module0_flow.util.units as units
import yaml
def get_detector_position(adc, channel, geometry_data):
    # Extract relevant data from the geometry data
    tpc_center = geometry_data['tpc_center_offset']
    det_center = geometry_data['det_center']
    det_adc_all = geometry_data['det_adc']  
    det_chan_all = geometry_data['det_chan'] 
    
    # Initialize variables to hold detector and tpc numbers
    detector_number = None
    tpc_number = None
    
    # Loop through all TPCs to find the one corresponding to the given channel and ADC
    for tpc in range(len(det_adc_all)):
        det_adc = det_adc_all[tpc]
        det_chan = det_chan_all[tpc]
        for det_num, adc_num in det_adc.items():
            if adc_num != adc:  # Skip if the ADC number doesn't match
                continue
            if channel in det_chan[det_num]:
                detector_number = det_num
                tpc_number = tpc
                break
                
    # If detector_number and tpc_number are still None, the channel was not found
    if detector_number is None or tpc_number is None:
        return None
    
    # Calculate 3D position
    _, y, z = det_center[int(detector_number)]
    x, _, _ = tpc_center[int(tpc_number)]
    #print(f'x: {x}, y: {y}, z: {z}')
    #print(det_center)
    return [x, y, z]
    
class WaveformHitFinder(H5FlowStage):
    '''
        Find waveforms from light events that surpass a simple threshold. This script is meant to run be run on filtered summed raw waveforms.

        Parameters:
         - ``sum_wvfm_dset_name``: ``str``, path to input filtered summed waveforms
         - ``hits_dset_name``: ``str``, path to output hits dataset
         - ``threshold``: ``dict`` of ``dict`` containing sets of ``tpc_index: {channel_index: threshold, ...}`` used for hit finding. A fixed global value can also be specified with a single ``float`` value

         ``sum_wvfm_dset_name`` is required in the cache.

         Requires RunData and Geometry resources in workflow.

         ``hits`` datatype::

            id                  u4,             unique identifier
            tpc                 u1,             tpc (for sum_hit)
            sum_chan            u1,             detector id
            trap_type           u2,             light detector type (LCM = 1, ACL = 0)
            boundary            f4(3),          (x,y,z) boundaries of det
            amplitude           f4,             peak adc value
            ts_pps              f4,             PPS timestamp of light event
            unix                i8,             UNIX timestamp of light event with second-level precision
    '''
    class_version = '0.0.0'

    default_hits_dset_name = 'light/simple_hits'
    default_LCM_threshold = 10
    default_ACL_threshold = 10
    default_single_module = -1
    
    def __init__(self, **params):
        super(WaveformHitFinder, self).__init__(**params)
        self.sum_wvfm_dset_name = params.get('sum_wvfm_dset_name')
        self.hits_dset_name = params.get('hits_dset_name', self.default_hits_dset_name)
        self.LCM_threshold = params.get('LCM_threshold', self.default_LCM_threshold)
        self.ACL_threshold = params.get('ACL_threshold', self.default_ACL_threshold)
        self.single_module = params.get('single_module', self.default_single_module)
        print(f"{self.single_module=}")
    def init(self, source_name):
        super(WaveformHitFinder, self).init(source_name)

        wvfm_dset = self.data_manager.get_dset(self.sum_wvfm_dset_name)
        self.nsamples = wvfm_dset.dtype['samples'].shape[2]

        self.hits_dtype = np.dtype([
                    ('id', 'u4'),
                    ('tpc', 'u1'),
                    ('sum_chan', 'u1'),
                    ('trap_type', 'u1'),
                    ('boundary', 'f4', (2,3)),
                    ('samples', 'f4', (self.nsamples,)),
                    ('amplitude', 'f4'),
                    ('ts_pps', 'f8'),
                    ('unix', 'i8')
                ])
        
        # create datasets and references
        self.data_manager.create_dset(self.hits_dset_name,
                                      dtype=self.hits_dtype)
        self.data_manager.create_ref(source_name, self.hits_dset_name)
        #self.data_manager.set_attrs(self.hits_dset_name,
        #                            classname=self.classname,
        #                            class_version=self.class_version,
        #                            wvfm_dset=self.sum_wvfm_dset_name
        #                            )

    def run(self, source_name, source_slice, cache):
        super(WaveformHitFinder, self).run(source_name, source_slice, cache)
        wvfms = np.array(cache[self.sum_wvfm_dset_name].reshape(cache[source_name].shape)[
            'samples'])  # 1:1 relationship
        events = np.array(cache[source_name])
        
        hits_data = np.zeros((0,), dtype=self.hits_dtype)
        hits_event_id = []

        events_tai_ns = []
        events_utime_ms = []
        nskipped_events = 0
        for i in range(len(events)):
            event = events[i]
            tai_ns = event['tai_ns'][event['tai_ns'] != 0]
            if not len(tai_ns):
                tai_ns = 0
            else:
                tai_ns = np.mean(tai_ns)*1e-3
            utime_ms = int(event['utime_ms'][event['utime_ms'] != 0][0]*1e-3)
            events_tai_ns.append(tai_ns)
            events_utime_ms.append(utime_ms)

        schan_to_trap_type_dict = {0: 1, 1: 0, 2: 1, 3: 0, \
                                  4: 1, 5: 0, 6: 1, 7: 0, \
                                  8: 1, 9: 0, 10: 1, 11: 0, \
                                  12: 1, 13: 0, 14: 1, 15: 0}

        if self.single_module in [1,2,3]:
            schan_to_adc_chan_list_dict = {0:[(1, 15),(1, 14),(1, 13),(1, 12),(1, 11),(1, 10)], \
                                      1: [(0, 15),(0, 14),(0, 13),(0, 12),(0, 11),(0, 10)],\
                                      2: [(1, 9),(1, 8),(1, 7),(1, 6),(1, 5),(1, 4)], \
                                      3: [(0, 9),(0, 8),(0, 7),(0, 6),(0, 5),(0, 4)], \
                                      4: [(1, 31),(1, 30),(1, 29),(1, 28),(1, 27),(1, 26)], \
                                      5: [(0, 31),(0, 30),(0, 29),(0, 28),(0, 27),(0, 26)], \
                                      6: [(1, 25),(1, 24),(1, 23),(1, 22),(1, 21),(1, 20)], \
                                      7: [(0, 25),(0, 24),(0, 23),(0, 22),(0, 21),(0, 20)], \
                                      8: [(1, 63),(1, 62),(1, 61),(1, 60),(1, 59),(1, 58)], \
                                      9: [(0, 63),(0, 62),(0, 61),(0, 60),(0, 59),(0, 58)], \
                                      10: [(1, 57),(1, 56),(1, 55),(1, 54),(1, 53),(1, 52)], \
                                      11: [(0, 57),(0, 56),(0, 55),(0, 54),(0, 53),(0, 52)], \
                                      12: [(1, 47),(1, 46),(1, 45),(1, 44),(1, 43),(1, 42)], \
                                      13: [(0, 47),(0, 46),(0, 45),(0, 44),(0, 43),(0, 42)], \
                                      14: [(1, 41),(1, 40),(1, 39),(1, 38),(1, 37),(1, 36)], \
                                      15: [(0, 41),(0, 40),(0, 39),(0, 38),(0, 37),(0, 36)]}
            with open('/global/cfs/cdirs/dune/users/sfogarty/ndlar_flow_low_energy/ndlar_flow/scripts/low_energy_scripts/2x2/ndlar_flow_low_energy_2/ndlar_flow/data/module1_flow/light_module_desc-0.2.0.yaml') as gf:
                        lrs_geometry_yaml = yaml.load(gf, Loader=yaml.FullLoader)
        elif self.single_module == 0:
            schan_to_adc_chan_list_dict = {0:[(0, 30),(0, 29),(0, 28),(0, 27),(0, 26),(0, 25)], \
                                      1: [(0, 23),(0, 22),(0, 21),(0, 20),(0, 19),(0, 18)],\
                                      2: [(0, 14),(0, 13),(0, 12),(0, 11),(0, 10),(0, 9)], \
                                      3: [(0, 7),(0, 6),(0, 5),(0, 4),(0, 3),(0, 2)], \
                                      4: [(1, 62),(1, 61),(1, 60),(1, 59),(1, 58),(1, 57)], \
                                      5: [(1, 55),(1, 54),(1, 53),(1, 52),(1, 51),(1, 50)], \
                                      6: [(1, 46),(1, 45),(1, 44),(1, 43),(1, 42),(1, 41)], \
                                      7: [(1, 39),(1, 38),(1, 37),(1, 36),(1, 35),(1, 34)], \
                                      8: [(1, 30),(1, 29),(1, 28),(1, 27),(1, 26),(1, 25)], \
                                      9: [(1, 23),(1, 22),(1, 21),(1, 20),(1, 19),(1, 18)], \
                                      10: [(1, 14),(1, 13),(1, 12),(1, 11),(1, 10),(1, 9)], \
                                      11: [(1, 7),(1, 6),(1, 5),(1, 4),(1, 3),(1, 2)], \
                                      12: [(0, 62),(0, 61),(0, 60),(0, 59),(0, 58),(0, 57)], \
                                      13: [(0, 55),(0, 54),(0, 53),(0, 52),(0, 51),(0, 50)], \
                                      14: [(0, 46),(0, 45),(0, 44),(0, 43),(0, 42),(0, 41)], \
                                      15: [(0, 39),(0, 38),(0, 37),(0, 36),(0, 35),(0, 34)]}
            with open('/global/cfs/cdirs/dune/users/sfogarty/ndlar_flow_low_energy/ndlar_flow/scripts/low_energy_scripts/2x2/ndlar_flow_low_energy_2/ndlar_flow/data/module0_flow/light_module_desc-0.0.0.yaml') as gf:
                        lrs_geometry_yaml = yaml.load(gf, Loader=yaml.FullLoader)
        if self.single_module != -1:
            boundaries_dict = {}
            for sum_chan in list(schan_to_adc_chan_list_dict.keys()):
                adc_chan_list = schan_to_adc_chan_list_dict[sum_chan]
                x_all, y_all, z_all = [],[],[]
                for adc_chan in adc_chan_list:
                    x,y,z = get_detector_position(adc_chan[0], adc_chan[1], lrs_geometry_yaml)
                    #if self.single_module == 2: #and not (((y < 31 and y > 0) or (y > -62 and y < -31)) and (x < 0)):
                    #    z = z*-1
                        
                    #if self.single_module == 2 and ((y > 31 and y < 62) and (x < 0) and (z > 0)):
                    #    y = 15.50975
                    #elif self.single_module == 2 and ((y > 0 and y < 31) and (x < 0) and (z > 0)):
                    #    y = 46.52925
                    x_all.append(x)
                    y_all.append(y)
                    z_all.append(z)
                boundaries_dict[sum_chan] = np.array([[min(x_all), min(y_all), min(z_all)], [max(x_all), max(y_all), max(z_all)]])
            
        max_of_wvfms = np.max(wvfms, axis=3)
        threshold = min(self.LCM_threshold, self.ACL_threshold)
        indices = np.where(max_of_wvfms > threshold) # preliminary thresholding... just to select hits to loop over
        for i, (event_index, tpc, sum_chan) in enumerate(zip(indices[0], indices[1], indices[2])):
            if self.single_module != -1:
                trap_type = schan_to_trap_type_dict[sum_chan]
                if self.single_module == 0 and trap_type == 0:
                    continue # skip mod0 ACL from single module run at Bern
                if trap_type == 0:
                    threshold = self.ACL_threshold
                else:
                    threshold = self.LCM_threshold
            max_of_wvfm = max_of_wvfms[event_index, tpc, sum_chan]
            
            if max_of_wvfm < threshold:
                continue
            hit_data = np.zeros((1,), dtype=self.hits_dtype)
            hit_data['tpc'] = tpc
            hit_data['sum_chan'] = sum_chan
            if self.single_module != -1:
                hit_data['trap_type'] = trap_type #resources['Geometry'].sum_chan_to_trap_type[(tpc, sum_chan)] # schan_to_trap_type_dict[sum_chan] #
            else:
                hit_data['trap_type'] = resources['Geometry'].sum_chan_to_trap_type[(tpc, sum_chan)] # schan_to_trap_type_dict[sum_chan]

            #adc_chan_list = schan_to_adc_chan_list_dict[sum_chan]

            
            #if sum_chan == 11:
            #    sum_chan_swap = 15
            #elif sum_chan == 15:
            #    sum_chan_swap = 11
            if self.single_module != -1:
                sum_chan_swap = sum_chan
                hit_data['boundary'] = boundaries_dict[sum_chan_swap] #resources['Geometry'].sum_chan_bounds[(tpc, sum_chan)]
            else:
                hit_data['boundary'] = resources['Geometry'].sum_chan_bounds[(tpc, sum_chan)]
            #boundaries_dict[sum_chan_swap]
            hit_data['samples'] = wvfms[event_index, tpc, sum_chan, :]
            hit_data['amplitude'] = max_of_wvfm
            hit_data['ts_pps'] = events_tai_ns[event_index]
            hit_data['unix'] = events_utime_ms[event_index]
            hits_data = np.concatenate((hits_data, hit_data))
            hits_event_id.append(events[event_index]['id'])

        hit_slice = self.data_manager.reserve_data(self.hits_dset_name, len(hits_data))
        hits_data['id'] = hit_slice.start + np.arange(len(hits_data), dtype=int)
        self.data_manager.write_data(self.hits_dset_name, hit_slice, hits_data)
        hits_event_id = np.array(hits_event_id)
        if len(hits_data):
            ref = np.c_[hits_event_id, hits_data['id']]
        else:
            ref = np.empty((0, 2))
        self.data_manager.write_ref(source_name, self.hits_dset_name, ref)
