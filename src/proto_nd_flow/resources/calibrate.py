import numpy as np
import yaml
import json
from collections import defaultdict

from h5flow.core import H5FlowResource
from h5flow.core import resources

import proto_nd_flow.util.units as units
from proto_nd_flow.util.compat import assert_compat_version

class Calibrate(H5FlowResource):
    '''
    Helper functions to do calibration of data products.

    Parameters:
         - ``path``: ``str``, path to stored calibration data within file
         - ``pedestal_file``: ``str``, path to yaml file containing channel-by-channel pedestal values
         - ``configuration_file``: ``str``, path to yaml file containing channel-by-channel vref/vcm values
         - ``pedestal_mv``: ``float``, default value for channel pedestal to apply to all channels unless pedestal_file is provided [mV]
         - ``vref_mv``: ``float``, default value for channel's vref to apply to all channels unless configuration_file is provided [mV]
         - ``vcm_mv``: ``float``, default value for channel's vcm to apply to all channels unless configuration_file is provided [mV]
         - ``adc_counts``: ``int``, dynamic range for ADC defined as 2^N where N is the number of bits in the ADC (defaults to 2**8)
         - ``gain``: ``float``, default channel gain [mV/ke-]

    Provides:
         - ``charge_from_dataword``: helper function for calculating charge in ke- from packet dataword
         - ``charge_from_dataword_corrected``: helper function that accounts for changes in vref, vcm due to nonlinearities 
                                               in adc (excessive load on vref/vcm bypass capacitors on tile PCB)
         - ``get_unique_ids``: helper function to calculate pixel unique ids for an array of packets
        
    Example usage::

         from h5flow.core import resources

         resources['Calibrate'].gain
         resources['Calibrate'].charge_from_dataword(packets)

    Example config::

         resources:
            - classname: CalibrateData
              params:
                path: 'calibrate_info'

    '''
    class_version = '0.0.0'
    default_path = 'calibrate_info'
    default_pedestal_file = ''
    default_configuration_file = ''
    default_pedestal_mv = 580
    default_vref_mv = 1568.0
    default_vcm_mv = 478.1
    default_adc_counts = 256
    default_gain = 4.522

    def __init__(self, **params):
        super(Calibrate, self).__init__(**params)

        self.path = params.get('path', self.default_path)
        self._pedestal_file = params.get('pedestal_file', self.default_pedestal_file)
        self._configuration_file = params.get('configuration_file', self.default_configuration_file)
        self._pedestal_mv = params.get('pedestal_mv', self.default_pedestal_mv)
        self._vref_mv = params.get('vref_mv', self.default_vref_mv)
        self._vcm_mv = params.get('vcm_mv', self.default_vcm_mv)
        self._adc_counts = params.get('adc_counts', self.default_adc_counts)
        self._gain = params.get('gain', self.default_gain)

        #: ASIC ADC configuration lookup table
        self._configuration = defaultdict(lambda: dict(
            vref_mv=self.default_vref_mv,
            vcm_mv=self.default_vcm_mv
        ))
        #: pixel pedestal value
        self._pedestal = defaultdict(lambda: dict(
            pedestal_mv=self.default_pedestal_mv
        ))
        
    def init(self, source_name):
        super(Calibrate, self).init(source_name)

        # create group (if not present)
        self.data_manager.set_attrs(self.path)
        # load data (if present)
        self.data = dict(self.data_manager.get_attrs(self.path))

        if not self.data:
            self._load_pedestals()
            self._load_configurations()

            self.data_manager.set_attrs(self.path,
                                        classname=self.classname,
                                        class_version=self.class_version,
                                        pedestal_file=self._pedestal_file,
                                        configuration_file=self._configuration_file,
                                        pedestal_mv=self._pedestal_mv,
                                        vref_mv=self._vref_mv,
                                        vcm_mv=self._vcm_mv,
                                        adc_counts=self._adc_counts,
                                        gain=self._gain
                                        )
        else:
            assert_compat_version(self.class_version, self.data['class_version'])

            self._pedestal_file = self.data['pedestal_file']
            self._configuration_file = self.data['configuration_file']
            self._pedestal_mv = self.data['pedestal_mv']
            self._vref_mv = self.data['vref_mv']
            self._vcm_mv = self.data['vcm_mv']
            self._adc_counts = self.data['adc_counts']
            self._gain = self.data['gain']

            self._load_pedestals()
            self._load_configurations()
                
    def _load_configurations(self):
        if self._configuration_file != '' and not resources['RunData'].is_mc:
            with open(self._configuration_file, 'r') as infile:
                for key, value in json.load(infile).items():
                    self._configuration[key] = value
        
    def _load_pedestals(self):
        if self._pedestal_file != '' and not resources['RunData'].is_mc:
            with open(self._pedestal_file, 'r') as infile:
                for key, value in json.load(infile).items():
                    self._pedestal[key] = value

    @staticmethod
    def _get_vref_vcm_correction(t, t_hits, tau_rc_vref = 4400.0, impulse_vref=-0.352, tau_rc_vcm=1460.0, impulse_vcm=0.352):
        return Calibrate._exp_sum( t, t_hits, impulse_vref, tau_rc_vref), Calibrate._exp_sum( t, t_hits, impulse_vcm, tau_rc_vcm  )

    @staticmethod
    def _exp_sum(t, ts, amps, taus):
        mask = ts <= t
        ts = ts[ mask ]
        if not type(amps) in [int, float, np.float64]:
            amps = amps[mask]
            taus = taus[mask]
        if np.sum(mask)==0:
            return 0
        dt = np.int64(t - ts)   # uint64 -> int64 (avoid OverflowError below)
        return np.sum( amps * np.exp( -1*dt/taus  )  )  
    
    def charge_from_dataword(self, packets):
        dw = packets['dataword']
        uid_arr = self.get_unique_ids(packets)
        vref, vcm, ped = [],[],[]
        if self._configuration_file != '':
            vref = np.array(
                    [self._configuration[unique_id]['vref_mv'] for unique_id in uid_arr.astype('str')])
            vcm = np.array(
                    [self._configuration[unique_id]['vcm_mv'] for unique_id in uid_arr.astype('str')])
        else:
            vref = np.full(len(uid_arr), self._vref_mv)
            vcm = np.full(len(uid_arr), self._vcm_mv)
        if self._pedestal_file != '':
            ped = np.array([self._pedestal[unique_id]['pedestal_mv'] for unique_id in uid_arr.astype('str')])
        else:
            ped = np.full(len(uid_arr), self._pedestal_mv)
        return (dw / self._adc_counts * (vref - vcm) + vcm - ped) / self._gain

    def charge_from_dataword_corrected(self, packets):
        '''
        Helper function to calculate charge in ke- from packet dataword 
        (accounts for changes in vref, vcm due to nonlinearities in adc (excessive load on vref/vcm bypass capacitors on tile PCB))

        :param packets: array of charge packets 
        :returns: ``unique ids array``

        '''
        dw = packets['dataword']
        ts = packets['timestamp']
        uid_arr = self.get_unique_ids(packets)
        vref, vcm = [],[]
        if self._configuration_file != '':
            vref = np.array(
                    [self._configuration[unique_id]['vref_mv'] for unique_id in uid_arr.astype('str')])
            vcm = np.array(
                    [self._configuration[unique_id]['vcm_mv'] for unique_id in uid_arr.astype('str')])
        else:
            vref = np.full(len(uid_arr), self._vref_mv)
            vcm = np.full(len(uid_arr), self._vcm_mv)
        if self._pedestal_file != '':
            ped = np.array([self._pedestal[unique_id]['pedestal_mv'] for unique_id in uid_arr.astype('str')])
        else:
            ped = np.full(len(uid_arr), self._pedestal_mv)
        # Find chips that had 
        chip_uid = (uid_arr.astype('int') // 100)*100
        chips, counts = np.unique(chip_uid, return_counts=True)
        
        vref_arr = np.full( dw.shape, vref  )
        vcm_arr =  np.full( dw.shape, vcm   )
        
        for chip in chips[counts > 1]:
     
            mask = chip_uid==chip

            chip_ts = ts[mask]

            #collect all vref, vcm corrections for hits on this chip
            vref_corrs = np.zeros( chip_ts.shape )
            vcm_corrs = np.zeros( chip_ts.shape )
            
            for ihit, t in enumerate(chip_ts):

                vref_corr, vcm_corr = self._get_vref_vcm_correction(t, chip_ts)

                vref_corrs[ihit] = vref_corr
                vcm_corrs[ihit] = vcm_corr

            vcm_arr[mask] += vcm_corrs
            vref_arr[mask] += vref_corrs
             
        return (dw / self._adc_counts * (vref_arr - vcm_arr) + vcm_arr - ped) / self._gain

    def get_unique_ids(self, packets):
        '''
        Helper function to get pixel unique ids for an array of charge packets. 

        :param packets: array of packets
        :returns: ``unique ids array``

        '''
        tile_id = resources['Geometry'].tile_id[packets['io_group'], packets['io_channel']]
        unique_ids = (packets['io_group'].astype(int)*1000_000_000
                            + tile_id.astype(int)*100_000
                            + packets['chip_id'].astype(int)*100
                            + packets['channel_id'].astype(int))
        return unique_ids
        
    @property
    def pedestal(self):
        '''
            Pedestal dictionary containing channel-by-channel pedestal values. Takes a unique_id as input.
        '''
        return self._pedestal

    @property
    def configuration(self):
        '''
            Configuration dictionary containing channel-by-channel vref and vcm values. Takes a unique_id as input.
        '''
        return self._configuration

    @property
    def vref_mv(self):
        '''
            Default vref_mv value used [mV]
        '''
        return self._vref_mv

    @property
    def vcm_mv(self):
        '''
            Default vcm_mv value used [mV]
        '''
        return self._vcm_mv
    
    @property
    def adc_counts(self):
        '''
            Dynamic range of the ADC in adc counts
        '''
        return self._adc_counts

    @property
    def gain(self):
        '''
            Pixel gain [mV/ke-]
        '''
        return self._gain
